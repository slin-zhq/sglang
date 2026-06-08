import atexit
import csv
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from sglang.srt.speculative import spec_cycle_logger

logger = logging.getLogger(__name__)

DARK_REQUIRED_COLUMNS = [
    "prompt_id",
    "block_id",
    "pos_j",
    "C_j",
    "D_j",
    "attn_input_max",
    "attn_self_jminus1",
    "attn_self_jplus1",
    "attn_self_jminus2",
    "depth",
    "top1_prob",
    "H_j",
    "accept_j",
]

DARK_ENABLED = False
ATTENTION_AVAILABLE = False

_csv_file = None
_csv_writer: Optional[csv.writer] = None
_lock = threading.Lock()
_warned_attention_unavailable = False


def default_dark_csv_path() -> str:
    env_path = os.environ.get("DARK_ATTENTION_CSV", "").strip()
    if env_path:
        return env_path
    return str(Path(spec_cycle_logger.LOG_PATH) / "dark_attention.csv")


def init_dark_logger(path: str) -> None:
    global DARK_ENABLED, _csv_file, _csv_writer
    output_path = path.strip() if path else default_dark_csv_path()
    with _lock:
        if _csv_file is not None:
            try:
                _csv_file.flush()
                _csv_file.close()
            except Exception:
                pass
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        _csv_file = open(output_path, "w", newline="", buffering=1)
        _csv_writer = csv.writer(_csv_file)
        _csv_writer.writerow(DARK_REQUIRED_COLUMNS)
        DARK_ENABLED = True


def mark_attention_unavailable() -> None:
    global ATTENTION_AVAILABLE, _warned_attention_unavailable
    ATTENTION_AVAILABLE = False
    if _warned_attention_unavailable:
        return
    _warned_attention_unavailable = True
    logger.warning(
        "DARK: attention weights not available from this backend; C_j/D_j logged as zeros. "
        "Use --attention-backend torch_native for real attention data."
    )


def current_prompt_and_block() -> tuple[int, int]:
    if spec_cycle_logger.ENABLED:
        spec_cycle_logger._refresh_log_path_from_control_file()
    context = getattr(spec_cycle_logger, "_record_context", {})
    prompt_id = int(context.get("question_id", -1))
    block_id = int(getattr(spec_cycle_logger, "_current_cycle_idx", -1))
    return prompt_id, block_id


def log_dark_row(
    prompt_id,
    block_id,
    pos_j,
    C_j,
    D_j,
    attn_input_max,
    attn_self_jminus1,
    attn_self_jplus1,
    attn_self_jminus2,
    depth,
    top1_prob,
    H_j,
    accept_j,
) -> None:
    if not DARK_ENABLED:
        return
    if int(prompt_id) == -1:
        return
    with _lock:
        if _csv_writer is None:
            return
        _csv_writer.writerow(
            [
                int(prompt_id),
                int(block_id),
                int(pos_j),
                float(C_j),
                float(D_j),
                float(attn_input_max),
                float(attn_self_jminus1),
                float(attn_self_jplus1),
                float(attn_self_jminus2),
                int(depth),
                float(top1_prob),
                float(H_j),
                int(accept_j),
            ]
        )


def close_dark_logger() -> None:
    global DARK_ENABLED, _csv_file, _csv_writer
    with _lock:
        if _csv_file is not None:
            try:
                _csv_file.flush()
                _csv_file.close()
            except Exception:
                pass
        _csv_file = None
        _csv_writer = None
        DARK_ENABLED = False


atexit.register(close_dark_logger)
