from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from chgnet.utils import (
    cuda_devices_sorted_by_free_mem,
    determine_device,
    gpu_devices_sorted_by_free_mem,
    solve_charge_by_mag,
)

if TYPE_CHECKING:
    from pymatgen.core import Structure


def test_gpu_devices_sorted_by_free_mem() -> None:
    # can't test this any better on CPU
    # but good to check it at least doesn't crash on CPU
    for device_type in ("cuda", "xpu"):
        lst = gpu_devices_sorted_by_free_mem(device_type=device_type)
        backend = getattr(torch, device_type, None)
        if backend is not None and backend.is_available():
            assert len(lst) > 0
            assert sorted(lst) == list(range(len(lst)))
        else:
            assert lst == []


def test_cuda_devices_sorted_by_free_mem() -> None:
    # deprecated alias, must keep matching the cuda branch
    assert cuda_devices_sorted_by_free_mem() == gpu_devices_sorted_by_free_mem(
        device_type="cuda"
    )


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_gpu_devices_are_valid_torch_indices(device_type: str) -> None:
    """Returned indices must be usable as torch device indices.

    They are derived from `device_count()`, so they follow the visible-device
    mask (`CUDA_VISIBLE_DEVICES` / `ZE_AFFINITY_MASK`) rather than physical
    device numbering. A vendor library that enumerates all physical devices
    would return out-of-range indices here.
    """
    backend = getattr(torch, device_type, None)
    if backend is None or not backend.is_available():
        pytest.skip(f"No {device_type} device")

    devices = gpu_devices_sorted_by_free_mem(device_type=device_type)
    assert sorted(devices) == list(range(backend.device_count()))


@pytest.mark.parametrize(
    ("cuda", "xpu", "mps", "expected"),
    [
        (True, False, False, "cuda"),
        (False, True, False, "xpu"),
        (False, False, True, "mps"),
        (False, False, False, "cpu"),
        (True, True, False, "cuda"),
    ],
)
def test_determine_device(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda: bool,
    xpu: bool,
    mps: bool,
    expected: str,
) -> None:
    """determine_device picks the right backend without needing the hardware."""
    monkeypatch.delenv("CHGNET_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    if getattr(torch, "xpu", None) is not None:
        monkeypatch.setattr(torch.xpu, "is_available", lambda: xpu)
    elif xpu:
        pytest.skip("torch build has no xpu module")

    assert determine_device() == expected


def test_determine_device_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHGNET_DEVICE", "cpu")
    assert determine_device() == "cpu"


@pytest.mark.parametrize("key", ["final_magmom", "magmom"])
def test_solve_charge_by_mag(li_mn_o2: Structure, key: str) -> None:
    assert li_mn_o2.charge == 0

    li_mn_o2.add_site_property(key, [0.5] * len(li_mn_o2))  # add unphysical magmoms
    # get charge-decorated structure
    struct_with_chg = solve_charge_by_mag(li_mn_o2)
    assert struct_with_chg.charge == -2  # expect unphysical charge
