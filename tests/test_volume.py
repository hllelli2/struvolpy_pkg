import numpy as np
import pytest
from struvolpy import Volume
import os
import importlib

try:
    importlib.import_module("TEMPy")

    has_tempy = True
except ImportError:
    has_tempy = False

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture(scope="session")
def read_mrc():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    return Volume.from_file(f"{current_dir}/../test_data/3407_reduced.mrc")


def test_origin(read_mrc):
    assert np.allclose(
        read_mrc.origin, np.array([[318.60995483, 303.83996582, 305.94998169]])
    )


def test_voxel(read_mrc):
    assert np.allclose(
        read_mrc.voxelsize, np.array([1.05499983, 1.05499983, 1.05499983])
    )
    assert np.allclose(read_mrc.voxelspacing, 1.054999828338623)


def test_dimensions(read_mrc):
    assert np.allclose(
        read_mrc.dimensions, np.array([56.96999073, 66.46498919, 75.95998764])
    )
    assert np.allclose(read_mrc.shape, np.array([72, 63, 54]))


@pytest.mark.skipif(not has_tempy, reason="TEMPy is not installed")
def test_to_TEMPy(read_mrc):
    try:
        tm = read_mrc.TEMPy_map
        Volume.from_TEMPy_map(tm)
    except:
        assert False
