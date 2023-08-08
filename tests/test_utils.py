from __future__ import annotations

import torch

from chgnet.utils import cuda_devices_sorted_by_free_mem


def test_cuda_devices_sorted_by_free_mem():
    # can't test this any better on CPU
    # but good to check it doesn't crash on CPU
    if torch.cuda.is_available() is False:
        assert torch.cuda.device_count() == 0
    else:
        assert len(cuda_devices_sorted_by_free_mem()) > 0
