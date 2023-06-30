from __future__ import annotations

import re
from typing import TYPE_CHECKING

from monty.io import reverse_readfile
from pymatgen.io.vasp.outputs import Oszicar, Vasprun

if TYPE_CHECKING:
    from pymatgen.core import Structure


def parse_vasp_dir(file_root):
    """Parse VASP output files into structures and labels
    By default, the magnetization is read from mag_x from VASP,
    plz modify the code if magnetization is for (y) and (z).

    Args:
        file_root: the directory of the VASP calculation outputs
    """
    try:
        oszicar = Oszicar(file_root + "/OSZICAR")
        vasprun_orig = Vasprun(file_root + "/vasprun.xml", exception_on_bad_xml=False)
        outcar_filename = file_root + "/OUTCAR"
    except Exception:
        oszicar = Oszicar(file_root + "/OSZICAR.gz")
        vasprun_orig = Vasprun(
            file_root + "/vasprun.xml.gz", exception_on_bad_xml=False
        )
        outcar_filename = file_root + "/OUTCAR.gz"

    charge = []
    mag_x = []
    mag_y = []
    mag_z = []
    header = []
    all_lines = []

    for line in reverse_readfile(outcar_filename):  # filename : self.filename
        clean = line.strip()
        all_lines.append(clean)

    all_lines.reverse()
    # For single atom systems, VASP doesn't print a total line, so
    # reverse parsing is very difficult
    read_charge = False
    read_mag_x = False
    read_mag_y = False  # for SOC calculations only
    read_mag_z = False
    mag_x_all = []
    ion_step_count = 0

    for clean in all_lines:
        if "magnetization (x)" in clean:
            ion_step_count += 1
        if read_charge or read_mag_x or read_mag_y or read_mag_z:
            if clean.startswith("# of ion"):
                header = re.split(r"\s{2,}", clean.strip())
                header.pop(0)
            else:
                m = re.match(r"\s*(\d+)\s+(([\d\.\-]+)\s+)+", clean)
                if m:
                    tokens = [float(token) for token in re.findall(r"[\d\.\-]+", clean)]
                    tokens.pop(0)
                    if read_charge:
                        charge.append(dict(zip(header, tokens)))
                    elif read_mag_x:
                        mag_x.append(dict(zip(header, tokens)))
                    elif read_mag_y:
                        mag_y.append(dict(zip(header, tokens)))
                    elif read_mag_z:
                        mag_z.append(dict(zip(header, tokens)))
                elif clean.startswith("tot"):
                    if ion_step_count == (len(mag_x_all) + 1):
                        mag_x_all.append(mag_x)
                    read_charge = False
                    read_mag_x = False
                    read_mag_y = False
                    read_mag_z = False
        if clean == "total charge":
            read_charge = True
            read_mag_x, read_mag_y, read_mag_z = False, False, False
        elif clean == "magnetization (x)":
            mag_x = []
            read_mag_x = True
            read_charge, read_mag_y, read_mag_z = False, False, False
        elif clean == "magnetization (y)":
            mag_y = []
            read_mag_y = True
            read_charge, read_mag_x, read_mag_z = False, False, False
        elif clean == "magnetization (z)":
            mag_z = []
            read_mag_z = True
            read_charge, read_mag_x, read_mag_y = False, False, False
        elif re.search("electrostatic", clean):
            read_charge, read_mag_x, read_mag_y, read_mag_z = (
                False,
                False,
                False,
                False,
            )

    if len(oszicar.ionic_steps) == len(mag_x_all):  ## unfinished VASP job
        print("Unfinished OUTCAR")
        mag_x_all = mag_x_all
    elif len(oszicar.ionic_steps) == (len(mag_x_all) - 1):  ## finished job
        mag_x_all.pop(-1)

    n_atoms = len(vasprun_orig.ionic_steps[0]["structure"])
    dataset = {
        "structure": [step["structure"] for step in vasprun_orig.ionic_steps],
        "uncorrected_total_energy": [
            step["e_0_energy"] for step in vasprun_orig.ionic_steps
        ],
        "energy_per_atom": [
            step["e_0_energy"] / n_atoms for step in vasprun_orig.ionic_steps
        ],
        "force": [step["forces"] for step in vasprun_orig.ionic_steps],
        "magmom": [[step["tot"] for step in j] for j in mag_x_all],
    }
    if "stress" in vasprun_orig.ionic_steps[0]:
        dataset["stress"] = [step["stress"] for step in vasprun_orig.ionic_steps]
    else:
        dataset["stress"] = None

    return dataset


def solve_charge_by_mag(
    structure: Structure,
    default_ox: dict[str, float] | None = None,
    ox_ranges: dict[str, dict[tuple[float, float], int]] | None = None,
):
    """Solve oxidation states by magmom.

    Args:
        structure: input pymatgen structure
        default_ox (dict[str, float]): default oxidation state for elements.
            Default = dict(Li=1, O=-2)
        ox_ranges (dict[str, dict[tuple[float, float], int]]): user-defined range to
            convert magmoms into formal valence.
            Example for Mn (Default):
                ("Mn": (
                    (0.5, 1.5): 2,
                    (1.5, 2.5): 3,
                    (2.5, 3.5): 4,
                    (3.5, 4.2): 3,
                    (4.2, 5): 2
                ))
    """
    ox_list = []
    solved_ox = True
    default_ox = default_ox or {"Li": 1, "O": -2}
    ox_ranges = ox_ranges or {
        "Mn": {(0.5, 1.5): 2, (1.5, 2.5): 3, (2.5, 3.5): 4, (3.5, 4.2): 3, (4.2, 5): 2}
    }

    mag = structure.site_properties.get(
        "final_magmom", structure.site_properties.get("magmom")
    )

    for idx, site in enumerate(structure):
        assigned = False
        if site.species_string in ox_ranges:
            for (minmag, maxmag), magox in ox_ranges[site.species_string].items():
                if mag[idx] >= minmag and mag[idx] < maxmag:
                    ox_list.append(magox)
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
    return None
