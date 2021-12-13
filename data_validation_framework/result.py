"""Util functions."""
import logging
import warnings
from copy import deepcopy

import pandas as pd

from data_validation_framework.util import report_missing_columns

L = logging.getLogger(__name__)


class ValidationResult(dict):
    """Class to represent a validation result as a dict.

    Some formatting is automatically executed during the dict creation:

    * if not given, the `ret_code` is set to `0` if `is_valid` is `True`, else it is set to `1`.
    * if the given `ret_code` is inconsistent with the given `is_valid` value, a ValueError is
        raised. The `ret_code` can have a value greater that 1 to mark a warning. In this case,
        the `is_valid` value can be either `True` or `False`.

    Args:
        is_valid (bool): Validation state.
        ret_code (int): The return code of the validation function.
        comment (str): A comment to explain why the validation failed.
        exception (str): The exception raised by the validation function if any.
    """

    def __init__(self, is_valid, ret_code=None, comment=None, exception=None, **kwargs):
        """ """
        if ret_code is None:
            if is_valid:
                ret_code = 0
            else:
                ret_code = 1
        else:
            if ret_code == 0 and not is_valid:
                raise ValueError("The 'is_valid' value must be True when 'ret_code' == 0.")
            if ret_code == 1 and is_valid:
                raise ValueError("The 'is_valid' value must be False when 'ret_code' == 1.")
            if ret_code not in [0, 1] and comment is None:
                warnings.warn("A comment should be set when the 'ret_code' is greater than 1.")

        super().__init__(
            is_valid=is_valid, ret_code=ret_code, comment=comment, exception=exception, **kwargs
        )


class ValidationResultSet(pd.DataFrame):
    """Class containing the results of a validation task.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        output_columns (dict): New columns to save results in ({<column name>: <default value>})
    """

    out_cols = {
        "is_valid": True,  # if we don't yet have an is_valid, we consider them all
        "ret_code": None,
        "comment": None,
        "exception": None,
    }

    def __init__(self, *args, output_columns=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Skip weird states due to Pandas and tqdm interactions
        data = kwargs.get("data", args[0] if len(args) > 0 else None)
        if not isinstance(
            data, (ValidationResultSet, pd.core.internals.managers.BlockManager)
        ) and not set(self.index).intersection(set(ValidationResultSet.out_cols.keys())):
            self.format_data(output_columns)

    @property
    def _constructor(self):
        """Pandas internal constructor property.

        See details here:
        https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-subclassing-pandas
        """
        return ValidationResultSet

    @staticmethod
    def _col_comparator(col_index):
        """Comparator used to sort columns.

        The known columns are sorted in the given order, then the unknown columns are sorted
        at the end. In the unknown columns, the strings are sorted first then the other types.
        """
        out_cols = {k: f"   {num}" for num, k in enumerate(ValidationResultSet.out_cols.keys())}
        order = col_index.map(
            lambda i: out_cols.get(i, f"{'~' * (3 if isinstance(i, str) else 6)}_{i}")
        )
        return order

    def format_data(self, output_columns):
        """Format dataframe."""
        out_cols = deepcopy(ValidationResultSet.out_cols)

        # Add and erase task specific columns
        if output_columns is not None:
            out_cols.update(output_columns)

        for col, data in out_cols.items():
            if col not in self.columns:
                self[col] = data

        # Fill missing ret_code values
        self.loc[(self["ret_code"].isnull()) & (self["is_valid"]), "ret_code"] = 0
        self.loc[(self["ret_code"].isnull()) & (~self["is_valid"]), "ret_code"] = 1

        # Check missing columns in inputs
        req_cols = list(out_cols.keys())

        no_missing_cols, missing_report, _ = report_missing_columns(self, req_cols)

        if not no_missing_cols:  # pragma: no cover
            raise ValueError(missing_report)

        # Check column data
        for col in ["is_valid"]:
            assert not self[col].isnull().any(), f"The '{col}' column must not have null value."

        # Sort columns and return the DataFrame
        self.sort_index(axis=1, key=ValidationResultSet._col_comparator, inplace=True)
