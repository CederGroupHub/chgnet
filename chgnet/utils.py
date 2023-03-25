from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from pymatgen.core import Structure


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
    """Computes the mean absolute error between prediction and target
    Parameters
    ----------
    prediction: Tensor (N, 1)
    target: Tensor (N, 1).
    """
    return torch.mean(torch.abs(target - prediction))


def read_json(fjson):
    """Args:
        fjson (str) - file name of json to read.

    Returns:
        dictionary stored in fjson
    """
    with open(fjson) as file:
        return json.load(file)


def write_json(d, fjson):
    """Args:
        d (dict) - dictionary to write
        fjson (str) - file name of json to write.

    Returns:
        written dictionary
    """
    with open(fjson, "w") as file:
        json.dump(d, file)


def solve_charge_by_mag(
    structure: Structure,
    default_ox: dict[str, float] = None,
    ox_ranges: dict[str, dict[tuple[float, float], int]] = None,
):
    """Solve oxidation states by magmom.

    Args:
        structure: input pymatgen structure
        default_ox (dict[str, float]): default oxidation state for elements.
            Default = {"Li": 1, "O": -2}
        ox_ranges (dict[str, dict[tuple[float, float], int]]): user defined range to
            convert magmoms into formal valence. Default = {
                "Mn": {(0.5, 1.5): 2, (1.5, 2.5): 3, (2.5, 3.5): 4, (3.5, 4.2): 3, (4.2, 5): 2}
            }
    """
    ox_list = []
    solved_ox = True
    default_ox = default_ox or {"Li": 1, "O": -2}
    ox_ranges = ox_ranges or {
        "Mn": {(0.5, 1.5): 2, (1.5, 2.5): 3, (2.5, 3.5): 4, (3.5, 4.2): 3, (4.2, 5): 2}
    }

    mag_key = (
        "final_magmom" if "final_magmom" in structure.site_properties else "magmom"
    )

    mag = structure.site_properties[mag_key]

    for site_i, site in enumerate(structure.sites):
        assigned = False
        if site.species_string in ox_ranges:
            for (minmag, maxmag), magox in ox_ranges[site.species_string].items():
                if mag[site_i] >= minmag and mag[site_i] < maxmag:
                    ox_list.append(magox)
                    # print(magox, mag[site_i])
                    assigned = True
                    break
        elif site.species_string in default_ox:
            ox_list.append(default_ox[site.species_string])
            assigned = True
        if not assigned:
            solved_ox = False

    if solved_ox:
        print(ox_list)
        structure.add_oxidation_state_by_site(ox_list)
        return structure

    else:
        return
