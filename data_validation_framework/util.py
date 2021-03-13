"""Util functions."""
import contextlib
import logging
import sys
import traceback
from itertools import repeat
from multiprocessing import Pool
from multiprocessing import Queue
from multiprocessing import current_process

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib import DummyTqdmFile

L = logging.getLogger(__name__)

tqdm.pandas()


@contextlib.contextmanager
def _std_out_err_redirect_tqdm():
    """Context manager used to redirect stdout to tqdm.write()."""
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:  # pragma: no cover
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


def _tqdm_wrapper(*args, **kwargs):
    _tqdm_wrapper._tqdm_queue.put(1)  # pylint: disable=protected-access
    return try_operation(*args, **kwargs)


def _init_tqdm_global_queue(queue):
    _tqdm_wrapper._tqdm_queue = queue  # pylint: disable=protected-access


def _apply_to_df_internal(data):
    df, args, kwargs = data
    process_name = current_process().name
    # pylint: disable=no-else-return
    if process_name != "MainProcess":
        return df.apply(_tqdm_wrapper, axis=1, args=args, **kwargs)
    else:
        return df.progress_apply(try_operation, axis=1, args=args, **kwargs)


def apply_to_df(df, func, nb_processes, *args, **kwargs):
    """Apply a function to df rows using tqdm."""
    if nb_processes is None or nb_processes == 1:
        # Serial computation
        return _apply_to_df_internal((df, ([func] + list(args)), kwargs))

    # Parallel computation
    nb_jobs = len(df)
    nb_chunks = min(nb_jobs, nb_processes)
    chunks = np.array_split(df, nb_chunks)
    queue = Queue()

    with _std_out_err_redirect_tqdm() as orig_stdout:
        with Pool(nb_chunks, _init_tqdm_global_queue, [queue]) as pool:
            # Create the progress bar
            progress_bar = tqdm(total=nb_jobs, file=orig_stdout, dynamic_ncols=True)

            # Start the computation
            results_list = pool.imap(
                _apply_to_df_internal,
                zip(chunks, repeat([func] + list(args)), repeat(kwargs)),
            )
            pool.close()

            # Update progress bar using the Queue
            for _ in range(nb_jobs):
                db = queue.get()
                progress_bar.update(db)

            # Wait for the last jobs to complete
            pool.join()

    # Gather all results
    all_res = pd.concat(results_list)
    return all_res


def try_operation(row, func, *args, **kwargs):
    """Try to apply a function on a :class:`pandas.Series`, and record exception."""
    if not row.is_valid:
        return row

    try:
        res = func(row, *args, **kwargs)
        row.loc[res.keys()] = list(res.values())
    except Exception:  # pylint: disable=broad-except
        exception = "".join(traceback.format_exception(*sys.exc_info()))
        L.warning("Exception for ID %s: %s", row.name, exception)
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
