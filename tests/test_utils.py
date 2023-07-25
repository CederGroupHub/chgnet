from __future__ import annotations

import torch

from chgnet.utils import cuda_devices_sorted_by_free_mem


def test_cuda_devices_sorted_by_free_mem():
    # can't test this any better on CPU
    # but good to check it doesn't crash on CPU
    assert torch.cuda.device_count() == 0
    assert cuda_devices_sorted_by_free_mem() == []
