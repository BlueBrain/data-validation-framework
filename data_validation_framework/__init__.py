"""data-validation-framework package.

Simple framework to create data validation workflows.
"""
import importlib.metadata

from data_validation_framework import result  # noqa
from data_validation_framework import task  # noqa

__version__ = importlib.metadata.version("data-validation-framework")
