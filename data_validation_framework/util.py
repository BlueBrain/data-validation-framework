"""Util functions."""
import logging
import sys
import traceback

from tqdm import tqdm

tqdm.pandas()

L = logging.getLogger(__name__)


def apply_to_df(df, func, *args, **kwargs):
    """Apply a function to df rows using tqdm."""
    return df.progress_apply(try_operation, axis=1, args=[func] + list(args), **kwargs)


def try_operation(row, func, *args, **kwargs):
    """Try to apply a function on a :class:`pandas.Series`, and record exception."""
    if not row.is_valid:
        return row

    try:
        res = func(row, *args, **kwargs)
        row.loc[res.keys()] = list(res.values())
    except Exception:  # pylint: disable=broad-except
        exception = "".join(traceback.format_exception(*sys.exc_info()))
        L.warning("Exception for combo %s: %s", row.name, exception)
        row.exception = exception
        row.is_valid = False
        row.ret_code = 1
    return row


def check_missing_columns(df, required_columns):
    """Return a list of missing columns in a :class:`pandas.DataFrame`.

    Args:
        df (pandas.DataFrame): The DataFrame to check.
        required_columns (list): The list of column names. A column name can be a str for a one
            level column or either a list(tuple(str)) or a dict(list(str)) for a two-level column.
    """
    missing_cols = []
    for col in required_columns:  # pylint: disable=too-many-nested-blocks
        if isinstance(col, str) and col not in df.columns.get_level_values(0):
            missing_cols.append(col)
        elif isinstance(col, (list, tuple)) and col not in df.columns:
            missing_cols.append(col)
        elif isinstance(col, dict):
            for level1 in col:
                if isinstance(col[level1], str):
                    col_level2 = [col[level1]]
                else:
                    col_level2 = col[level1]
                if level1 not in df.columns.get_level_values(0):
                    for level2 in col_level2:
                        missing_cols.append((level1, level2))
                else:
                    for level2 in col_level2:
                        if level2 not in df[level1].columns.get_level_values(0):
                            missing_cols.append((level1, level2))
    return sorted(set(missing_cols))


def report_missing_columns(df, required_columns):
    """Check that required columns exist in a :class:`pandas.DataFrame`.

    Args:
        df (pandas.DataFrame): The DataFrame to check.
        required_columns (list): The list of column names. A column name can be a str for a one
            level column or either a list(tuple(str)) or a dict(list(str)) for a two-level column.
    """
    missing_cols = check_missing_columns(df, required_columns)
    report_str = f"Missing columns: {sorted(missing_cols)}"
    if missing_cols:
        return False, report_str, missing_cols
    return True, report_str, missing_cols
