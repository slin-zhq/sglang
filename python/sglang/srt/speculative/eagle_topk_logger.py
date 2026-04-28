"""
eagle_topk_logger.py
────────────────────
Per-cycle data logger for the torch.topk optimisation experiment.

Environment variables controlling behaviour:
  EAGLE_TOPK_EXP_LOG_ENABLE    - "1" to enable all logging (default: "1")
  EAGLE_TOPK_EXP_LOG_PATH      - absolute directory where JSONL files are written.
                                 The benchmark script sets this per config/dataset/rep/turn.
                                 Default: /mnt/zhiqi/sglang_eagle3_optimize/outputs/torch_topk_optimization
  EAGLE_ATTN_MS_LOG_ENABLE     - "1" to additionally instrument the FlashInfer
                                 TARGET_VERIFY attention call with torch.cuda.Event
                                 and emit `attn_ms` in timing.jsonl. Default: "0".
                                 Subordinate to EAGLE_TOPK_EXP_LOG_ENABLE — if that
                                 is 0, attn_ms logging is also disabled.

When EAGLE_TOPK_EXP_LOG_ENABLE != "1", every function in this module
returns immediately with zero overhead.

Logging granularity
───────────────────
One JSONL line = one speculative-decoding cycle for one request in the batch.
Fields are documented in experiment_plan.md Section 7.

Each record also carries the benchmark context:
    bench_name, rep_idx, question_id, turn_id

These fields are injected by the benchmark harness before each generate() call.
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
ATTN_MS_ENABLED: bool = os.environ.get("EAGLE_ATTN_MS_LOG_ENABLE", "0") == "1"
LOG_PATH: str = os.environ.get("EAGLE_TOPK_EXP_LOG_PATH", _DEFAULT_LOG_PATH)
CONTROL_PATH: str = os.environ.get("EAGLE_TOPK_EXP_LOG_CONTROL_PATH", "")

# ──────────────────────────────────────────────
# Thread-safe state
# ──────────────────────────────────────────────
_lock = threading.Lock()
_cycle_counter: int = 0          # global cycle index, incremented once per decode cycle
_current_cycle_idx: int = 0      # idx for the in-flight cycle; shared by all log functions
_record_context = {
    "bench_name": "",
    "rep_idx": -1,
    "question_id": -1,
    "turn_id": -1,
}
_control_mtime: Optional[float] = None
_write_counters: Dict[str, int] = {}
_decode_counters: Dict[str, int] = {}  # counts log_timing calls per timing.jsonl path


# File handles keyed by their absolute path so we can reuse open handles
_open_files: Dict[str, object] = {}

# Pending (start_event, end_event) pairs from the TARGET_VERIFY attention path.
# Drained once per cycle in log_timing() when ATTN_MS_ENABLED.
_pending_attn_events: List[tuple] = []


def accumulate_attn_event(evt_start, evt_end) -> None:
    """
    Record a (start, end) CUDA event pair from one TARGET_VERIFY attention call.

    Called from FlashInferAttnBackend.forward_extend around the
    prefill_wrapper_paged.forward invocation, once per transformer layer per
    decode cycle (32 calls per cycle for Llama 3.1 8B).

    Both events must have been .record()-ed on the current CUDA stream before
    this function returns. No GPU sync happens here — sync is deferred to
    _drain_attn_events_ms() called from log_timing().

    When ATTN_MS_ENABLED is False, callers skip the wrap entirely; this
    function is never invoked.
    """
    if not (ENABLED and ATTN_MS_ENABLED):
        return
    _pending_attn_events.append((evt_start, evt_end))


def _drain_attn_events_ms() -> float:
    """
    Sum the elapsed time across all pending (start, end) event pairs, in ms.

    Assumes the caller has already called torch.cuda.synchronize() (log_timing
    does so via _sync_and_time()). Clears the pending list after reading.
    """
    if not _pending_attn_events:
        return 0.0
    total_ms = 0.0
    for s, e in _pending_attn_events:
        try:
            total_ms += s.elapsed_time(e)
        except Exception:
            # Event not yet recorded (e.g., raced), skip rather than crash.
            continue
    _pending_attn_events.clear()
    return total_ms


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
            control_payload = f.read().strip()
    except Exception:
        return

    _control_mtime = mtime
    if not control_payload:
        return

    new_path = control_payload
    new_context = None

    if control_payload.startswith("{"):
        try:
            payload = json.loads(control_payload)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            new_path = str(payload.get("log_path", "")).strip()
            if new_path:
                new_context = {
                    "bench_name": str(payload.get("bench_name", "")),
                    "rep_idx": int(payload.get("rep_idx", -1)),
                    "question_id": int(payload.get("question_id", -1)),
                    "turn_id": int(payload.get("turn_id", -1)),
                }

    if new_path and new_path != LOG_PATH:
        set_log_path(new_path)

    if new_context is not None:
        set_record_context(
            new_context["bench_name"],
            new_context["rep_idx"],
            new_context["question_id"],
            new_context["turn_id"],
        )


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
    global LOG_PATH, _cycle_counter, _current_cycle_idx, _write_counters, _decode_counters, _open_files
    with _lock:
        # Close any open file handles from the previous run
        for fh in _open_files.values():
            try:
                fh.close()
            except Exception:
                pass
        _open_files = {}
        _write_counters = {}
        _decode_counters = {}
        _cycle_counter = 0
        _current_cycle_idx = 0
        LOG_PATH = path


def set_record_context(
    bench_name: str,
    rep_idx: int,
    question_id: int,
    turn_id: int,
) -> None:
    """Set the benchmark identity fields that are embedded into each JSONL row."""
    global _record_context
    with _lock:
        _record_context = {
            "bench_name": bench_name,
            "rep_idx": int(rep_idx),
            "question_id": int(question_id),
            "turn_id": int(turn_id),
        }


def _base_record_fields() -> dict:
    """Return the shared identity fields for the current logging context."""
    return dict(_record_context)


def _get_file_handle(filename: str):
    """Return (and cache) an open file handle for `filename` inside LOG_PATH."""
    full_path = os.path.join(LOG_PATH, filename)
    if full_path not in _open_files:
        Path(full_path).parent.mkdir(parents=True, exist_ok=True)
        _open_files[full_path] = open(full_path, "a", buffering=1)  # line-buffered
    return _open_files[full_path]


def _write_jsonl(filename: str, record: dict) -> None:
    full_path = os.path.join(LOG_PATH, filename)
    with _lock:
        record_to_write = dict(record)
        record_to_write["record_idx"] = _write_counters.get(full_path, 0)
        _write_counters[full_path] = record_to_write["record_idx"] + 1
        fh = _get_file_handle(filename)
        fh.write(json.dumps(record_to_write, default=_json_default) + "\n")


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
    Called from two sites:
      1. ``eagle_draft_cuda_graph_runner.replay()`` — CUDA-graph path, reads from
         pre-allocated debug buffers written during graph execution.
      2. ``eagle_worker.EAGLEWorker.draft_forward()`` — eager path (``--disable-cuda-graph``),
         reads directly from the local tensors already computed in that function.

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
            **_base_record_fields(),
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
            **_base_record_fields(),
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
        _write_jsonl("verify.jsonl", record)


def log_prefill_timing(prefill_ms: float) -> None:
    """
    Called from eagle_worker.forward_target_extend() to record the duration of
    the first-cycle target-model extend (prefill) pass.

    Writes one row to timing.jsonl with event="prefill".  This is always the
    very first row in the file (record_idx=0 / cold_start=True) and is written
    before any decode-cycle rows.  Callers that only want steady-state timing
    should filter on event=="timing" (not "prefill").
    """
    if not ENABLED or _is_cuda_graph_capturing():
        return

    _refresh_log_path_from_control_file()

    record = {
        **_base_record_fields(),
        "cycle_idx": -1,   # not a decode cycle
        "event": "prefill",
        "cold_start": True,
        "prefill_ms": round(prefill_ms, 4),
    }
    _write_jsonl("timing.jsonl", record)


def log_timing(
    cycle_idx: int,
    timings: dict,
) -> None:
    """
    Called from eagle_worker.forward_batch_generation() decode branch.
    Logs phase-level timings for Experiment 1.

    `timings` must have keys draft_ms, verify_ms, extend_ms, cycle_ms with
    values already in milliseconds (caller is responsible for the conversion).

    Sets cold_start=True on the first call per turn (tracked via
    _decode_counters) regardless of any prefill row that may precede it.
    """
    if not ENABLED or _is_cuda_graph_capturing():
        return

    _refresh_log_path_from_control_file()

    full_path = os.path.join(LOG_PATH, "timing.jsonl")
    # cold_start is True for the first *decode* row in each turn.
    # _decode_counters tracks how many log_timing() calls have been made for
    # this path, independently of _write_counters (which also counts the
    # leading prefill row emitted by log_prefill_timing()).  This ensures
    # cold_start=True on the first decode row regardless of whether a prefill
    # row was written first.
    # NOTE: We acquire-then-release the lock *before* calling _write_jsonl,
    # which also acquires the same (non-reentrant) lock.  Holding the lock
    # across _write_jsonl would deadlock.
    with _lock:
        decode_idx = _decode_counters.get(full_path, 0)
        _decode_counters[full_path] = decode_idx + 1

    record = {
        **_base_record_fields(),
        "cycle_idx": cycle_idx,
        "event": "timing",
        "cold_start": decode_idx == 0,
        **timings,  # caller passes ms values directly
    }

    # Drain TARGET_VERIFY attention events accumulated during this cycle.
    # _sync_and_time() at the end of the verify phase has already called
    # torch.cuda.synchronize(), so elapsed_time() is safe here.
    if ATTN_MS_ENABLED:
        record["attn_ms"] = round(_drain_attn_events_ms(), 4)

    _write_jsonl("timing.jsonl", record)


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
