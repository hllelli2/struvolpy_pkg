import numpy as np
import pytest
from struvolpy import Structure
from constants import ROTATED_COORDINATES_0, RESIDUES, WEIGHTS
import os
import warnings
import importlib

try:
    importlib.import_module("TEMPy")

    has_tempy = True
except ImportError:
    has_tempy = False

warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture(scope="session")
def read_pdb():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    return Structure.from_file(f"{current_dir}/../test_data/Bchain.pdb")


@pytest.mark.skipif(not has_tempy, reason="TEMPy is not installed")
def test_gemmi_to_tempy(read_pdb):
    try:
        read_pdb.to_TEMPy()
    except:
        assert False


def test_rotate(read_pdb):
    # 90 degree rotation about the x-axis, 30 degree rotation about the z-axis, 25 degree rotation about the y-axis
    rotmat = np.array([[1, 0, 0], [0, 0.8660254, -0.5], [0, 0.5, 0.8660254]])
    read_pdb.rotate(rotmat)
    assert np.allclose(read_pdb.coor[0], ROTATED_COORDINATES_0, atol=1e-3)


def test_translate(read_pdb):
    translation = np.array([4, 16, 64])
    startx, starty, startz = (
        read_pdb.coor[0][0],
        read_pdb.coor[1][0],
        read_pdb.coor[2][0],
    )

    read_pdb.translate(translation)
    assert np.all(
        (
            read_pdb.coor[0][0] == startx + translation[0],
            read_pdb.coor[1][0] == starty + translation[1],
            read_pdb.coor[2][0] == startz + translation[2],
        )
    )


def test_weights(read_pdb):
    # will also test get_atoms
    assert np.allclose(read_pdb.weights, WEIGHTS, atol=1e-5)


def test_get_residues(read_pdb):
    assert read_pdb.get_residues() == RESIDUES


def test_duplicate(read_pdb):
    new_pdb = read_pdb.duplicate()
    assert id(new_pdb) != id(read_pdb)
