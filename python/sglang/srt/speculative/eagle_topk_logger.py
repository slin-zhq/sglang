"""
eagle_topk_logger.py
────────────────────
Per-cycle data logger for the torch.topk optimisation experiment.

Two environment variables control behaviour:
  EAGLE_TOPK_EXP_LOG_ENABLE  - "1" to enable (default: "1")
  EAGLE_TOPK_EXP_LOG_PATH    - absolute directory where JSONL files are written.
                                The benchmark script sets this per config/dataset/rep/turn.
                                Default: /mnt/zhiqi/sglang_eagle3_optimize/outputs/torch_topk_optimization

When disabled (EAGLE_TOPK_EXP_LOG_ENABLE != "1"), every function in this module
returns immediately with zero overhead.

Logging granularity
───────────────────
One JSONL line = one speculative-decoding cycle for one request in the batch.
Fields are documented in experiment_plan.md Section 7.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

# ──────────────────────────────────────────────
# Configuration (read once at module import)
# ──────────────────────────────────────────────
_DEFAULT_LOG_PATH = (
    "/mnt/zhiqi/sglang_eagle3_optimize/outputs/torch_topk_optimization"
)

ENABLED: bool = os.environ.get("EAGLE_TOPK_EXP_LOG_ENABLE", "1") == "1"
LOG_PATH: str = os.environ.get("EAGLE_TOPK_EXP_LOG_PATH", _DEFAULT_LOG_PATH)
CONTROL_PATH: str = os.environ.get("EAGLE_TOPK_EXP_LOG_CONTROL_PATH", "")

# ──────────────────────────────────────────────
# Thread-safe state
# ──────────────────────────────────────────────
_lock = threading.Lock()
_cycle_counter: int = 0          # global cycle index, incremented once per decode cycle
_current_cycle_idx: int = 0      # idx for the in-flight cycle; shared by all log functions
_control_mtime: Optional[float] = None

# File handles keyed by their absolute path so we can reuse open handles
_open_files: Dict[str, object] = {}


def _is_cuda_graph_capturing() -> bool:
    """Return True when the current CUDA stream is inside graph capture."""
    if not torch.cuda.is_available():
        return False
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    if is_capturing is None:
        return False
    try:
        return bool(is_capturing())
    except Exception:
        return False


# ──────────────────────────────────────────────
# Timing helpers
# ──────────────────────────────────────────────
def _sync_and_time() -> float:
    """Return wall-clock seconds, after synchronising CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _refresh_log_path_from_control_file() -> None:
    """
    Refresh LOG_PATH from a control file shared across processes.

    This allows the benchmark parent process to switch turn directories while
    scheduler child processes keep logging to the latest path.
    """
    global _control_mtime
    if not CONTROL_PATH:
        return

    try:
        mtime = os.path.getmtime(CONTROL_PATH)
    except Exception:
        return

    if _control_mtime is not None and mtime <= _control_mtime:
        return

    try:
        with open(CONTROL_PATH, "r") as f:
            new_path = f.read().strip()
    except Exception:
        return

    _control_mtime = mtime
    if new_path and new_path != LOG_PATH:
        set_log_path(new_path)


# ──────────────────────────────────────────────
# Cycle management
# ──────────────────────────────────────────────
def begin_cycle() -> int:
    """
    Mark the start of a new speculative-decoding decode cycle.

    Atomically increments the global cycle counter and stores the snapshot
    as _current_cycle_idx.  All three log functions (log_timing,
    log_organize_draft_results, log_verify_result) read _current_cycle_idx
    so every file written in the same cycle shares the same cycle_idx.

    Call exactly once per decode cycle from
    eagle_worker.EAGLEWorker.forward_batch_generation(), before draft().

    Returns the cycle_idx for this cycle.
    """
    global _cycle_counter, _current_cycle_idx
    with _lock:
        idx = _cycle_counter
        _cycle_counter += 1
        _current_cycle_idx = idx
    return idx


# ──────────────────────────────────────────────
# File helpers
# ──────────────────────────────────────────────
def set_log_path(path: str) -> None:
    """
    Update the log path at runtime (called by the benchmark script before each run).
    Also resets the cycle counter so per-run counters start from 0.
    """
    global LOG_PATH, _cycle_counter, _current_cycle_idx, _open_files
    with _lock:
        # Close any open file handles from the previous run
        for fh in _open_files.values():
            try:
                fh.close()
            except Exception:
                pass
        _open_files = {}
        _cycle_counter = 0
        _current_cycle_idx = 0
        LOG_PATH = path


def _get_file_handle(filename: str):
    """Return (and cache) an open file handle for `filename` inside LOG_PATH."""
    full_path = os.path.join(LOG_PATH, filename)
    if full_path not in _open_files:
        Path(full_path).parent.mkdir(parents=True, exist_ok=True)
        _open_files[full_path] = open(full_path, "a", buffering=1)  # line-buffered
    return _open_files[full_path]


def _write_jsonl(filename: str, record: dict) -> None:
    fh = _get_file_handle(filename)
    fh.write(json.dumps(record, default=_json_default) + "\n")


def _json_default(obj):
    """Fallback JSON serialiser for non-standard types."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return str(obj)


# ──────────────────────────────────────────────
# Public logging API
# ──────────────────────────────────────────────

def log_organize_draft_results(
    score_list_flat: torch.Tensor,    # (bs, total_candidates) – all cum-log-probs
    top_scores_indices: torch.Tensor, # (bs, num_draft_token-1) – selected indices
    top_scores_values: torch.Tensor,  # (bs, num_draft_token-1) – selected scores
    all_token_ids: torch.Tensor,      # (bs, total_candidates) – all token IDs
    num_draft_token: int,
) -> None:
    """
    Called from eagle_draft_cuda_graph_runner.replay() after the CUDA graph
    replay returns, using the debug buffers written during graph execution.
    Logs the full candidate pool and what topk selected.

    cycle_idx is read from _current_cycle_idx, which was set by begin_cycle()
    at the top of the same decode cycle in eagle_worker.forward_batch_generation().
    """
    if not ENABLED or _is_cuda_graph_capturing():
        return

    _refresh_log_path_from_control_file()

    cycle_idx = _current_cycle_idx

    bs = score_list_flat.shape[0]
    scores_cpu = score_list_flat.detach().cpu()
    topk_idx_cpu = top_scores_indices.detach().cpu()
    topk_val_cpu = top_scores_values.detach().cpu()
    tokens_cpu = all_token_ids.detach().cpu()

    for b in range(bs):
        record = {
            "cycle_idx": cycle_idx,
            "batch_elem": b,
            "event": "organize_draft_results",
            "num_candidates": int(scores_cpu.shape[1]),
            "num_draft_token": num_draft_token,
            "all_scores": scores_cpu[b].tolist(),
            "all_token_ids": tokens_cpu[b].tolist(),
            "topk_selected_indices": topk_idx_cpu[b].tolist(),
            "topk_selected_scores": topk_val_cpu[b].tolist(),
        }
        _write_jsonl("organize_draft.jsonl", record)


def log_verify_result(
    candidates: torch.Tensor,     # (bs, draft_token_num)
    target_predict: torch.Tensor, # (bs, draft_token_num)
    accept_index: torch.Tensor,   # (bs, spec_steps+1)
    accept_length: torch.Tensor,  # (bs,)
    predict: torch.Tensor,        # accepted token IDs
) -> None:
    """
    Called from eagle_info.EagleVerifyInput.verify() after acceptance checking.
    Logs ground-truth acceptance data needed for Experiment 3 & 4.

    cycle_idx is read from _current_cycle_idx, which was set by begin_cycle()
    at the top of the same decode cycle in eagle_worker.forward_batch_generation().
    """
    if not ENABLED or _is_cuda_graph_capturing():
        return

    _refresh_log_path_from_control_file()

    cycle_idx = _current_cycle_idx

    bs = candidates.shape[0]
    candidates_cpu = candidates.detach().cpu()
    target_cpu = target_predict.detach().cpu()
    acc_idx_cpu = accept_index.detach().cpu()
    acc_len_cpu = accept_length.detach().cpu()
    predict_cpu = predict.detach().cpu().reshape(-1)

    for b in range(bs):
        acc_len = int(acc_len_cpu[b].item())
        record = {
            "cycle_idx": cycle_idx,
            "batch_elem": b,
            "event": "verify",
            "accept_length": acc_len,
            "candidates": candidates_cpu[b].tolist(),
            "target_model_argmax": target_cpu[b].tolist(),
            "accept_index": acc_idx_cpu[b].tolist(),
            "accepted_tokens": (
                predict_cpu[acc_idx_cpu[b, :acc_len]].tolist() if acc_len > 0 else []
            ),
        }
        _write_jsonl("verify_oracle.jsonl", record)


def log_timing(
    cycle_idx: int,
    timings: dict,
    accept_length_per_req: List[int],
) -> None:
    """
    Called from eagle_worker.forward_batch_generation() decode branch.
    Logs phase-level timings and per-request accept lengths for Experiment 1.

    `timings` must have keys draft_ms, verify_ms, extend_ms, cycle_ms with
    values already in milliseconds (caller is responsible for the conversion).
    """
    if not ENABLED or _is_cuda_graph_capturing():
        return

    _refresh_log_path_from_control_file()

    record = {
        "cycle_idx": cycle_idx,
        "event": "timing",
        **timings,  # caller passes ms values directly
        "accept_length_per_req": accept_length_per_req,
        "mean_accept_length": (
            sum(accept_length_per_req) / len(accept_length_per_req)
            if accept_length_per_req else 0.0
        ),
    }
    _write_jsonl("cycle_data.jsonl", record)


def flush_all() -> None:
    """Flush all open file handles. Call at the end of each question."""
    if not ENABLED:
        return
    with _lock:
        for fh in _open_files.values():
            try:
                fh.flush()
            except Exception:
                pass


def close_all() -> None:
    """Close all open file handles. Call at the end of each run."""
    global _open_files
    with _lock:
        for fh in _open_files.values():
            try:
                fh.close()
            except Exception:
                pass
        _open_files = {}
