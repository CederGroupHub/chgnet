from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from chgnet.utils import cuda_devices_sorted_by_free_mem, solve_charge_by_mag

if TYPE_CHECKING:
    from pymatgen.core import Structure


def test_cuda_devices_sorted_by_free_mem():
    # can't test this any better on CPU
    # but good to check it at least doesn't crash on CPU
    lst = cuda_devices_sorted_by_free_mem()
    if torch.cuda.is_available():
        assert len(lst) > 0
    else:
        assert lst == []


@pytest.mark.parametrize("key", ["final_magmom", "magmom"])
def test_solve_charge_by_mag(li_mn_o2: Structure, key: str) -> None:
    assert li_mn_o2.charge == 0

    li_mn_o2.add_site_property(key, [0.5] * len(li_mn_o2))  # add unphysical magmoms
    # get charge-decorated structure
    struct_with_chg = solve_charge_by_mag(li_mn_o2)
    assert struct_with_chg.charge == -2  # expect unphysical charge
