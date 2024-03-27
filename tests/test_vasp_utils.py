from __future__ import annotations

from typing import TYPE_CHECKING
from zipfile import ZipFile

import pytest
from pymatgen.core import Structure

from chgnet import ROOT
from chgnet.utils import parse_vasp_dir

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_vasp_dir_with_magmoms(tmp_path: Path):
    with ZipFile(f"{ROOT}/tests/files/parse-vasp-with-magmoms.zip") as zip_ref:
        zip_ref.extractall(tmp_path)
    dataset_dict = parse_vasp_dir(tmp_path)

    assert isinstance(dataset_dict, dict)
    assert len(dataset_dict["structure"]) > 0
    assert len(dataset_dict["uncorrected_total_energy"]) > 0
    assert len(dataset_dict["energy_per_atom"]) > 0
    assert len(dataset_dict["force"]) > 0
    assert len(dataset_dict["magmom"]) > 0
    assert len(dataset_dict["stress"]) > 0

    for structure in dataset_dict["structure"]:
        assert isinstance(structure, Structure)

    for magmom in dataset_dict["magmom"]:
        assert len(magmom) == len(dataset_dict["structure"][0])


def test_parse_vasp_dir_without_magmoms(tmp_path: Path):
    # using test.zip shared for error repro in
    # https://github.com/CederGroupHub/chgnet/issues/147
    with ZipFile(f"{ROOT}/tests/files/parse-vasp-no-magmoms.zip") as zip_ref:
        zip_ref.extractall(tmp_path)
    dataset_dict = parse_vasp_dir(tmp_path)

    assert isinstance(dataset_dict, dict)
    assert len(dataset_dict["structure"]) > 0
    assert len(dataset_dict["uncorrected_total_energy"]) > 0
    assert len(dataset_dict["energy_per_atom"]) > 0
    assert len(dataset_dict["force"]) > 0
    assert len(dataset_dict["magmom"]) > 0
    assert len(dataset_dict["stress"]) > 0

    for structure in dataset_dict["structure"]:
        assert isinstance(structure, Structure)

    for magmom in dataset_dict["magmom"]:
        assert len(magmom) == len(dataset_dict["structure"][0])
        assert all(mag == 0.0 for mag in magmom)


def test_parse_vasp_dir_no_data():
    # test non-existing directory
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        parse_vasp_dir(f"{ROOT}/tests/files/non-existent")

    # test existing directory without VASP files
    with pytest.raises(RuntimeError, match="No data parsed from"):
        parse_vasp_dir(f"{ROOT}/tests/files")
