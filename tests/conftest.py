"""Prepare the tests."""
# pylint: disable=missing-function-docstring
import os
from pathlib import Path

import pytest

from data_validation_framework.target import OutputLocalTarget

DATA = Path(__file__).parent / "data"


@pytest.fixture()
def tmp_working_dir(tmp_path):
    """Change working directory before a test and change it back when the test is finished."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(cwd)


@pytest.fixture()
def data_dir():
    """Path to the directory where the data are stored."""
    return DATA


@pytest.fixture(autouse=True)
def reset_target_prefix(tmpdir):
    """Automatically set the default prefix to the current test directory."""
    OutputLocalTarget.set_default_prefix(tmpdir)
    yield
    OutputLocalTarget.set_default_prefix(None)
