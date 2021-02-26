"""Specific targets."""
from luigi_tools.target import OutputLocalTarget


class TaggedOutputLocalTarget(OutputLocalTarget):
    """Target with tagged output path."""


class ReportTarget(TaggedOutputLocalTarget):
    """Specific target for BaseValidationTask reports."""

    def __init__(self, *args, task_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
