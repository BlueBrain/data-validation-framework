"""This module provides the main interface of the data-validation-framework package."""
from pkg_resources import get_distribution

from data_validation_framework import result  # noqa
from data_validation_framework import task  # noqa

__version__ = get_distribution("data-validation-framework").version
