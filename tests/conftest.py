"""Prepare the tests."""
import os
from pathlib import Path

import pytest

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
