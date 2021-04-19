"""Util functions."""
import contextlib
import logging
import multiprocessing
import queue
import sys
import threading
import traceback
from itertools import repeat
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

L = logging.getLogger(__name__)


class StreamToQueue(DummyTqdmFile):
    """Fake file-like stream object that redirects all prints to a Queue."""

    def __init__(self, *args, message_queue=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_queue = message_queue

    def write(self, buf):
        """Redirect write calls to the Queue."""
        if len(buf.rstrip()) > 0:
            self.message_queue.put(buf)


@contextlib.contextmanager
def _std_out_err_redirect_func(*args, **kwargs):
    """Context manager used to redirect stdout and stderr to tqdm.write()."""
    orig_out_err = sys.stdout, sys.stderr
    try:
        if _std_out_err_redirect_func._redirect:  # pylint: disable=protected-access
            sys.stdout = StreamToQueue(sys.stdout, *args, **kwargs)
            sys.stderr = StreamToQueue(sys.stderr, *args, **kwargs)
        yield
    # Relay exceptions
    except Exception as exc:  # pragma: no cover
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


def _tqdm_wrapper(*args, **kwargs):
    # pylint: disable=protected-access
    _tqdm_wrapper._tqdm_queue.put(1)
    with _std_out_err_redirect_func(message_queue=_tqdm_wrapper._message_queue):
        res = try_operation(*args, **kwargs)
    return res


def _init_tqdm_global_queue(tqdm_queue, message_queue, redirect=True):
    _tqdm_wrapper._tqdm_queue = tqdm_queue  # pylint: disable=protected-access
    _tqdm_wrapper._message_queue = message_queue  # pylint: disable=protected-access
    _std_out_err_redirect_func._redirect = redirect  # pylint: disable=protected-access


def _reset_tqdm_global_queue():
    _tqdm_wrapper._tqdm_queue = None  # pylint: disable=protected-access
    _tqdm_wrapper._message_queue = None  # pylint: disable=protected-access
    _std_out_err_redirect_func._redirect = True  # pylint: disable=protected-access


def _apply_to_df_internal(data):
    (num, df), args, kwargs = data
    return num, df.apply(_tqdm_wrapper, axis=1, args=args, **kwargs)


def tqdm_worker(progress_bar, tqdm_queue):
    """Update progress bar using the Queue."""
    while True:
        db = tqdm_queue.get()
        if db is None:
            return
        progress_bar.update(db)


def message_worker(progress_bar, message_queue):
    """Write a message without interfering with the progress bar using the message Queue."""
    while True:
        message = message_queue.get()
        if message is None:
            return
        progress_bar.write(message)


def apply_to_df(df, func, *args, nb_processes=None, redirect_stdout=None, **kwargs):
    """Apply a function to df rows using tqdm."""
    nb_jobs = len(df)
    if redirect_stdout is None:
        redirect_stdout = True

    if nb_processes is None or nb_processes <= 1:
        # Serial computation
        is_parallel = False
        tqdm_queue = queue.Queue()
        message_queue = queue.Queue()
    else:
        # Parallel computation
        is_parallel = True
        nb_chunks = min(nb_jobs, nb_processes)
        chunks = enumerate(np.array_split(df, nb_chunks))
        tqdm_queue = multiprocessing.Queue()
        message_queue = multiprocessing.Queue()

    # Create the progress bar
    progress_bar = tqdm(total=nb_jobs, dynamic_ncols=True)
    tqdm_thread = threading.Thread(target=tqdm_worker, args=(progress_bar, tqdm_queue))
    message_thread = threading.Thread(target=message_worker, args=(progress_bar, message_queue))

    tqdm_thread.start()
    message_thread.start()

    if is_parallel:
        with Pool(
            nb_chunks,
            _init_tqdm_global_queue,
            [tqdm_queue, message_queue, redirect_stdout],
        ) as pool:

            # Start the computation
            results_list = pool.imap(
                _apply_to_df_internal,
                zip(chunks, repeat([func] + list(args)), repeat(kwargs)),
            )
            pool.close()

            # Wait for the last jobs to complete
            pool.join()

        # Gather all results
        all_res = pd.concat([j for i, j in sorted(results_list, key=lambda x: x[0])])

    else:
        _init_tqdm_global_queue(tqdm_queue, message_queue, redirect_stdout)
        _, all_res = _apply_to_df_internal(((0, df), [func] + list(args), kwargs))

    _reset_tqdm_global_queue()

    # Terminate the threads
    tqdm_queue.put_nowait(None)
    message_queue.put_nowait(None)
    tqdm_thread.join()
    message_thread.join()

    # Close the progress bar
    progress_bar.close()

    # Return the results
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
