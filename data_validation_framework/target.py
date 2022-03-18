"""Specific targets."""
import warnings

from luigi_tools.target import OutputLocalTarget


class TaggedOutputLocalTarget(OutputLocalTarget):
    """Target with tagged output path."""

    _already_changed = False

    @classmethod
    def set_default_prefix(cls, prefix, quiet=False):
        """Set the default prefix of the class."""
        if cls._already_changed and not quiet and str(prefix) != str(cls.get_default_prefix()):
            warnings.warn(
                f"The default prefix of '{cls.__name__}' was already changed before, changing it "
                f"again might lead to unexpected behavior (old value: {cls.get_default_prefix()} ; "
                f"new value: {prefix}."
            )
        cls._already_changed = True
        super().set_default_prefix(prefix)


class _WithTaskNameMixin:
    """Mixin to add the task name as attribute."""

    def __init__(self, *args, task_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name


class ReportTarget(_WithTaskNameMixin, TaggedOutputLocalTarget):
    """Specific target for BaseValidationTask reports."""


class DataDirectoryTarget(_WithTaskNameMixin, TaggedOutputLocalTarget):
    """Specific target for BaseValidationTask reports."""
