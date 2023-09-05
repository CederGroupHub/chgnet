from __future__ import annotations

import re
from typing import TYPE_CHECKING

from monty.io import reverse_readfile
from pymatgen.io.vasp.outputs import Oszicar, Vasprun

if TYPE_CHECKING:
    from pymatgen.core import Structure


def parse_vasp_dir(
    file_root: str, check_electronic_convergence: bool = True
) -> dict[str, list]:
    """Parse VASP output files into structures and labels
    By default, the magnetization is read from mag_x from VASP,
    plz modify the code if magnetization is for (y) and (z).

    Args:
        file_root (str): the directory of the VASP calculation outputs
        check_electronic_convergence (bool): if set to True, this function will raise
            Exception to VASP calculation that did not achieve
    """
    try:
        oszicar = Oszicar(f"{file_root}/OSZICAR")
        vasprun_orig = Vasprun(
            f"{file_root}/vasprun.xml",
            parse_dos=False,
            parse_eigen=False,
            parse_projected_eigen=False,
            parse_potcar_file=False,
            exception_on_bad_xml=False,
        )
        outcar_filename = f"{file_root}/OUTCAR"
    except Exception:
        oszicar = Oszicar(f"{file_root}/OSZICAR.gz")
        vasprun_orig = Vasprun(
            f"{file_root}/vasprun.xml.gz",
            parse_dos=False,
            parse_eigen=False,
            parse_projected_eigen=False,
            parse_potcar_file=False,
            exception_on_bad_xml=False,
        )
        outcar_filename = f"{file_root}/OUTCAR.gz"

    charge = []
    mag_x = []
    mag_y = []
    mag_z = []
    header = []
    all_lines = []

    for line in reverse_readfile(outcar_filename):
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

    if len(oszicar.ionic_steps) == len(mag_x_all):  # unfinished VASP job
        print("Unfinished OUTCAR")
    elif len(oszicar.ionic_steps) == (len(mag_x_all) - 1):  # finished job
        mag_x_all.pop(-1)

    n_atoms = len(vasprun_orig.ionic_steps[0]["structure"])

    dataset = {
        "structure": [],
        "uncorrected_total_energy": [],
        "energy_per_atom": [],
        "force": [],
        "magmom": [],
        "stress": None if "stress" not in vasprun_orig.ionic_steps[0] else [],
    }

    for ionic_step, mag_step in zip(vasprun_orig.ionic_steps, mag_x_all):
        if (
            check_electronic_convergence
            and len(ionic_step["electronic_steps"]) >= vasprun_orig.parameters["NELM"]
        ):
            continue

        dataset["structure"].append(ionic_step["structure"])
        dataset["uncorrected_total_energy"].append(ionic_step["e_0_energy"])
        dataset["energy_per_atom"].append(ionic_step["e_0_energy"] / n_atoms)
        dataset["force"].append(ionic_step["forces"])
        dataset["magmom"].append([site["tot"] for site in mag_step])
        if "stress" in ionic_step:
            dataset["stress"].append(ionic_step["stress"])

    if dataset["uncorrected_total_energy"] == []:
        raise Exception(f"No data parsed from {file_root}!")

    return dataset


def solve_charge_by_mag(
    structure: Structure,
    default_ox: dict[str, float] | None = None,
    ox_ranges: dict[str, dict[tuple[float, float], int]] | None = None,
) -> Structure | None:
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
            for (min_mag, max_mag), mag_ox in ox_ranges[site.species_string].items():
                if mag[idx] >= min_mag and mag[idx] < max_mag:
                    ox_list.append(mag_ox)
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
