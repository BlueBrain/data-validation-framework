"""Test the data_validation_framework.target module."""
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
import pytest

from data_validation_framework import target


@pytest.fixture
def reset_target_no_prefix():
    """Reset the default prefix of the OutputLocalTarget class."""
    target.OutputLocalTarget.set_default_prefix(None)
    yield
    target.OutputLocalTarget.set_default_prefix(None)


def test_ReportTarget(reset_target_no_prefix):
    """Test the ReportTarget class."""
    report_target = target.ReportTarget("a_path", create_parent=False)
    assert report_target.task_name is None
    assert report_target.path == "a_path"

    report_target_with_name = target.ReportTarget(
        "a_path", task_name="test_name", create_parent=False
    )
    assert report_target_with_name.task_name == "test_name"
    assert report_target_with_name.path == "a_path"


@pytest.fixture
def reset_target_relative_prefix():
    """Set the default prefix of the OutputLocalTarget class."""
    target.OutputLocalTarget.set_default_prefix("parent_default_prefix")
    yield
    target.OutputLocalTarget.set_default_prefix(None)


def test_ReportTarget_with_parent_prefix(reset_target_relative_prefix):
    """Test the ReportTarget class when a default prefix is given to TaggedOutputLocalTarget."""
    report_target = target.ReportTarget("a_path", create_parent=False)
    assert report_target.path == "parent_default_prefix/a_path"

    report_target = target.ReportTarget("a_path", prefix="report_prefix", create_parent=False)
    assert report_target.path == "parent_default_prefix/report_prefix/a_path"

    try:
        target.ReportTarget.set_default_prefix("report_default_prefix")

        report_target = target.ReportTarget("a_path", create_parent=False)
        assert report_target.path == "parent_default_prefix/report_default_prefix/a_path"

        report_target = target.ReportTarget("a_path", prefix="report_prefix", create_parent=False)
        assert report_target.path == "parent_default_prefix/report_prefix/a_path"
    finally:
        target.ReportTarget.set_default_prefix(None)

    report_target_with_name = target.ReportTarget(
        "a_path", task_name="test_name", create_parent=False
    )
    assert report_target_with_name.task_name == "test_name"
