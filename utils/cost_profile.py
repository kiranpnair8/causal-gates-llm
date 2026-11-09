"""Optional wall-clock and CUDA cost reporting for calibration experiments."""

import json
import time
from pathlib import Path

import torch


def start_timer(reset_cuda_peak=False):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        if reset_cuda_peak:
            torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def stop_timer(start):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


def cuda_memory():
    if not torch.cuda.is_available():
        return {"gpu": None, "peak_memory_allocated_gib": None,
                "peak_memory_reserved_gib": None}
    device = torch.cuda.current_device()
    gib = 1024 ** 3
    return {"gpu": torch.cuda.get_device_name(device),
            "peak_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
            "peak_memory_reserved_gib": torch.cuda.max_memory_reserved(device) / gib}


def write_cost(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print("\nCost profile:")
    print(json.dumps(payload, indent=2))
    print(f"Saved cost profile to {path}")
