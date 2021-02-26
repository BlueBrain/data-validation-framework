"""Test the data_validation_framework.target module."""
from data_validation_framework import target


def test_ReportTarget():
    """Test the ReportTarget class."""
    report_target = target.ReportTarget("a path")
    assert report_target.task_name is None

    report_target_with_name = target.ReportTarget("a path", task_name="test_name")
    assert report_target_with_name.task_name == "test_name"
