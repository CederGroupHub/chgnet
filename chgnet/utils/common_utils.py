from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch import Tensor


def _default_accelerator() -> str:
    """Return the name of the best available torch accelerator backend.

    Returns:
        str: "cuda", "xpu" or "cpu" depending on what torch can see.
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def determine_device(
    use_device: str | None = None,
    *,
    check_cuda_mem: bool = False,
) -> str:
    """Determine the device to use for torch model.

    Args:
        use_device (str): User specify device name
        check_cuda_mem (bool): Whether to return the accelerator with the most
            available memory. Applies to both CUDA and XPU devices. Falls back
            to an unindexed device when free memory cannot be queried, see
            `gpu_devices_sorted_by_free_mem`. Default = False

    Returns:
        device (str): device name to be passed to model.to(device)
    """
    use_device = use_device or os.getenv("CHGNET_DEVICE")
    if use_device in {"mps", None} and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = use_device or _default_accelerator()
        if check_cuda_mem and device in {"cuda", "xpu"}:
            devices_by_mem = gpu_devices_sorted_by_free_mem(device_type=device)
            if devices_by_mem:
                device = f"{device}:{devices_by_mem[-1]}"

    return device


def gpu_devices_sorted_by_free_mem(device_type: str | None = None) -> list[int]:
    """List available GPU devices sorted by increasing available memory.

    Free memory is queried through torch itself (`mem_get_info`), so no
    vendor-specific management library is needed. Indices are torch device
    indices, so they respect `CUDA_VISIBLE_DEVICES` / `ZE_AFFINITY_MASK` and
    can be passed straight to `.to(device)`.

    To get the device with the most free memory, use the last list item.

    Backend caveats:

    - CUDA: `torch.cuda.mem_get_info` calls `cudaMemGetInfo` and reports
      device-wide free memory, matching what NVML reported.
    - XPU: `torch.xpu.mem_get_info` requires torch >= 2.6 and reads the SYCL
      `ext_intel_free_memory` device query. On some multi-tile Intel GPUs
      that query is reported per card rather than per tile, so the ranking
      can be coarse. Returned indices are always valid either way.

    An empty list is returned when free memory cannot be queried, in which
    case callers should fall back to an unindexed device.

    Args:
        device_type (str): "cuda" or "xpu". If None, the available accelerator
            is auto-detected. Default = None

    Returns:
        list[int]: GPU device indices sorted by increasing free memory.
    """
    device_type = device_type or _default_accelerator()
    backend = getattr(torch, device_type, None)

    if backend is None or not backend.is_available():
        return []
    # mps has no mem_get_info, and torch only added it for xpu in 2.6
    if not hasattr(backend, "mem_get_info"):
        return []

    free_memories = [
        backend.mem_get_info(idx)[0] for idx in range(backend.device_count())
    ]
    return sorted(range(len(free_memories)), key=lambda idx: free_memories[idx])


def cuda_devices_sorted_by_free_mem() -> list[int]:
    """List available CUDA devices sorted by increasing available memory.

    Deprecated alias kept for backward compatibility, prefer
    `gpu_devices_sorted_by_free_mem`.

    Returns:
        list[int]: CUDA device numbers sorted by increasing free memory.
    """
    return gpu_devices_sorted_by_free_mem(device_type="cuda")


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        """Initialize the meter."""
        self.reset()

    def reset(self) -> None:
        """Reset the meter value, average, sum and count to 0."""
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        """Update the meter value, average, sum and count.

        Args:
            val (float): New value to be added to the running average.
            n (int, optional): Number of times the value is added. Default = 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count != 0:
            self.avg = self.sum / self.count


def mae(prediction: Tensor, target: Tensor) -> Tensor:
    """Computes the mean absolute error between prediction and target.

    Args:
        prediction: Tensor (N, 1)
        target: Tensor (N, 1).

    Returns:
        tensor
    """
    return torch.mean(torch.abs(target - prediction))


def read_json(filepath: str) -> dict:
    """Read the JSON file.

    Args:
        filepath (str): file name of JSON to read.

    Returns:
        dict: data stored in filepath
    """
    with open(filepath) as file:
        return json.load(file)


def write_json(dct: dict, filepath: str) -> dict:
    """Write the JSON file.

    Args:
        dct (dict): dictionary to write
        filepath (str): file name of JSON to write.
    """

    def handler(obj: object) -> int | float | list | object:
        """Convert numpy types to JSON serializable types.

        Fixes TypeError: Object of type int64 is not JSON serializable
        reported in https://github.com/CederGroupHub/chgnet/issues/168.

        Returns:
            int | float | list | object: object for serialization
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(filepath, mode="w") as file:
        json.dump(dct, file, default=handler)


def mkdir(path: str) -> str:
    """Make directory.

    Args:
        path (str): directory name

    Returns:
        path
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("Folder exists")
    return path
