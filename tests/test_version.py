"""Test the version of the data-validation-framework package."""
import pkg_resources

import data_validation_framework as dvf


def test_version():
    """Test the version of the data-validation-framework package."""
    pkg_version = pkg_resources.get_distribution("data-validation-framework").version
    assert dvf.__version__ == pkg_version
