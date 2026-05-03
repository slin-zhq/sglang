import math
from enum import IntEnum
from typing import List, Optional, Tuple

import torch
from sglang.srt.utils import is_cuda, is_hip, is_npu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

if _is_cuda or _is_hip:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


_BUDGET_TECHNIQUES = ("baseline", "keep_shallow_full", "uniform_capped", "oracle_accept_prior")


def compute_depth_budget_vector(
    depth: int,
    k: int,
    candidate_budget: int,
    technique: str,
) -> Optional[List[int]]:
    """
    Compute the per-depth allocation vector for a given technique, once at startup.

    Returns None for "baseline" (use global topK unchanged).
    For all other techniques returns a list of length `depth` summing to
    `candidate_budget`, where entry d gives the number of candidates to keep
    at depth d+1 (0-indexed).

    Tree structure assumed (standard EAGLE3 d-k config):
      depth 1 pool = k
      depth d >= 2 pool = k * k  (only top-k from previous level is expanded)

    Techniques
    ----------
    keep_shallow_full
        Fill depths 1..D in order until budget is exhausted; distribute
        any remainder equally across the remaining deeper depths.
        This is the E7-validated near-optimal static policy.

    uniform_capped
        Allocate candidate_budget // depth slots per depth, capped by pool
        size; push leftover to the deepest depths.

    oracle_accept_prior
        Allocate proportional to 1/(d+1) prior (favours shallow), then cap
        and renormalise.
    """
    if technique == "baseline":
        return None

    if technique not in _BUDGET_TECHNIQUES:
        raise ValueError(
            f"Unknown budget allocation technique: {technique!r}. "
            f"Valid options: {_BUDGET_TECHNIQUES}"
        )

    # Pool sizes: depth-1 has k nodes; depth 2+ each have k² nodes
    pools: List[int] = [k] + [k * k] * max(0, depth - 1)

    if technique == "keep_shallow_full":
        b: List[int] = [0] * depth
        rem = candidate_budget
        last_full = 0
        for d in range(depth):
            if rem >= pools[d]:
                b[d] = pools[d]
                rem -= pools[d]
                last_full = d + 1
            else:
                break  # Do NOT assign rem here; distribute below across all remaining depths.
        # Distribute remaining slots evenly across unfilled depths.
        deeper = depth - last_full
        if deeper > 0 and rem > 0:
            per, extra = divmod(rem, deeper)
            for d in range(last_full, depth):
                b[d] = min(pools[d], per + (1 if extra > 0 else 0))
                if extra > 0:
                    extra -= 1

    elif technique == "uniform_capped":
        target = candidate_budget // depth
        b = [min(pools[d], target) for d in range(depth)]
        rem = candidate_budget - sum(b)
        for d in range(depth - 1, -1, -1):
            room = pools[d] - b[d]
            if room > 0 and rem > 0:
                take = min(room, rem)
                b[d] += take
                rem -= take

    elif technique == "oracle_accept_prior":
        weights = [1.0 / (d + 2) for d in range(depth)]  # 1/(d+1) for 1-indexed depth
        wsum = sum(weights)
        b = [min(pools[d], max(0, round(candidate_budget * w / wsum))) for d, w in enumerate(weights)]
        diff = candidate_budget - sum(b)
        # Adjust rounding residual from deepest up
        for d in range(depth - 1, -1, -1):
            if diff == 0:
                break
            if diff > 0:
                room = pools[d] - b[d]
                take = min(room, diff)
                b[d] += take
                diff -= take
            else:
                cut = min(b[d], -diff)
                b[d] -= cut
                diff += cut

    # Safety: cap by pool and ensure exact sum
    b = [min(b[d], pools[d]) for d in range(depth)]
    deficit = candidate_budget - sum(b)
    d = depth - 1
    while deficit != 0 and d >= 0:
        if deficit > 0:
            room = pools[d] - b[d]
            if room > 0:
                take = min(room, deficit)
                b[d] += take
                deficit -= take
        else:
            cut = min(b[d], -deficit)
            b[d] -= cut
            deficit += cut
        d -= 1

    return b


# --- Precomputed depth-slice layout (host-side, module-level cache) -----------
# Maps (depth, k) -> (starts, ends, pools) tuples so repeated calls within the
# same run never recompute these constants.
_depth_slice_cache: dict = {}


def _get_depth_slices(depth: int, k: int) -> Tuple[List[int], List[int], List[int]]:
    """Return (start, end, pool_size) for each depth level (0-indexed)."""
    key = (depth, k)
    if key not in _depth_slice_cache:
        starts, ends, pools = [], [], []
        for d in range(depth):
            if d == 0:
                s, e = 0, k
            else:
                s = k + (d - 1) * k * k
                e = s + k * k
            starts.append(s)
            ends.append(e)
            pools.append(e - s)
        _depth_slice_cache[key] = (starts, ends, pools)
    return _depth_slice_cache[key]


def _depth_aware_topk_indices(
    score_list_flat: torch.Tensor,
    depth_budget: List[int],
    k: int,
    depth: int,
) -> torch.Tensor:
    """
    Per-depth topK selection.  Returns a (bs, sum(depth_budget)) index tensor
    sorted by index value (ascending), matching the contract of the global topK path.

    The draft tree is stored level-by-level in score_list_flat:
      depth 1 → columns [0, k)
      depth d >= 2 → columns [k + (d-2)*k², k + (d-1)*k²)

    Ancestor closure is maintained because:
      - keep_shallow_full fills all lower depths completely, so every node at
        a deep depth has its entire ancestor chain selected.
      - For partial-fill depths, the within-depth topK naturally selects the
        highest-scoring candidates; their ancestors are already included by the
        lower-depth full coverage.

    CUDA-graph compatibility
    ─────────────────────────
    The previous implementation iterated over depths in a Python for-loop and
    conditionally appended tensors, which prevents CUDA graph capture (the graph
    records a static op sequence; host-side branching on runtime values is
    illegal inside a captured graph).

    This rewrite eliminates all data-dependent Python control flow:
    - Depth-slice boundaries (starts/ends) are constant for a given (depth, k)
      config and are precomputed once at module level.
    - Per-depth topK is expressed as a fixed sequence of tensor ops: a large
      negative sentinel is written into non-selected slots so that the global
      topK on each depth slice picks exactly the right candidates.
    - The final gather + sort operates on tensors whose shapes are fixed at
      graph capture time (total_select = sum(depth_budget) is a compile-time
      constant for a given config).

    Result: the function is safe to call inside a captured CUDA graph.
    """
    bs = score_list_flat.shape[0]
    device = score_list_flat.device
    total_cols = score_list_flat.shape[1]

    starts, ends, pools = _get_depth_slices(depth, k)

    # Build a "masked" score tensor where each depth slot retains its score
    # only for the top-b_d positions within that depth window, and is set to
    # -inf elsewhere.  This uses fixed-shape ops (no conditionals on values).
    masked = torch.full_like(score_list_flat, float("-inf"))

    for d_idx in range(depth):
        b_d = depth_budget[d_idx] if d_idx < len(depth_budget) else 0
        n_select = min(b_d, pools[d_idx])
        if n_select <= 0:
            continue  # static: n_select is a compile-time constant per config
        s, e = starts[d_idx], ends[d_idx]
        depth_scores = score_list_flat[:, s:e]           # (bs, pool_d)
        # topk on fixed-size slice — shape is static: (bs, n_select)
        _, local_idx = torch.topk(depth_scores, n_select, dim=-1, largest=True, sorted=False)
        # Scatter the real scores back into the masked tensor at selected positions.
        # global_idx shape: (bs, n_select) — still fixed.
        global_idx = local_idx + s
        # Use scatter to write scores; scatter_ is graph-safe (no Python loop over bs).
        masked.scatter_(1, global_idx, depth_scores.gather(1, local_idx))

    # Collect all non-(-inf) positions: there are exactly sum(n_selects) of them
    # (constant at graph-capture time).  We use topk on the full masked row.
    total_select = sum(
        min(depth_budget[d] if d < len(depth_budget) else 0, pools[d])
        for d in range(depth)
    )
    if total_select <= 0:
        return torch.zeros((bs, 0), dtype=torch.long, device=device)

    # topk with k=total_select on the masked row picks exactly the kept slots.
    _, all_selected = torch.topk(masked, total_select, dim=-1, largest=True, sorted=False)
    # Sort ascending to match the contract (index-order consistency with global topK path).
    return torch.sort(all_selected, dim=-1).values


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
    depth_budget: Optional[List[int]] = None,
    k: Optional[int] = None,
    depth: Optional[int] = None,
):
    """
    Select the top (num_draft_token - 1) candidates from the draft pool.

    When depth_budget is None (default / baseline), uses the original global
    torch.topk over all candidates.  When depth_budget is provided, uses
    per-depth selection via _depth_aware_topk_indices().
    """
    score_list = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)

    if depth_budget is not None and k is not None and depth is not None:
        top_scores_index = _depth_aware_topk_indices(score_list, depth_budget, k, depth)
    else:
        top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)
        top_scores_index = torch.sort(top_scores.indices).values

    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

    return parent_list, top_scores_index, draft_tokens


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


def build_tree_kernel_efficient(
    verified_id: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask = torch.full(
            (num_verify_tokens * bs * num_verify_tokens,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask = torch.full(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs,
            ),
            True,
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrive_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_index, retrive_next_token, retrive_next_sibling = retrive_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    if position_buf is not None:
        positions = position_buf
    else:
        positions = torch.empty(
            (bs * num_verify_tokens,), device=device, dtype=torch.long
        )

    if _is_npu:
        torch.ops.npu.build_tree_kernel_efficient(
            parent_list.to(dtype=torch.int64),
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    else:
        sgl_build_tree_kernel_efficient(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    return (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    )


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    if _is_cuda or _is_hip:
        from sgl_kernel import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )

    elif _is_npu:
        from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )
    return predicts, accept_index, accept_token_num
