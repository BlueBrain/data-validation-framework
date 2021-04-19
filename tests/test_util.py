"""Test the data_validation_framework.util module."""
# pylint: disable=missing-function-docstring
import os
import re
import time

import pandas as pd
import pytest

from data_validation_framework import result
from data_validation_framework import util


def test_report_missing_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    res, report_str, missing_cols = util.report_missing_columns(df, ["a"])
    assert res
    assert report_str == "Missing columns: []"
    assert missing_cols == []

    res, report_str, missing_cols = util.report_missing_columns(df, ["a", "b"])
    assert res
    assert report_str == "Missing columns: []"
    assert missing_cols == []

    res, report_str, missing_cols = util.report_missing_columns(df, ["c"])
    assert not res
    assert report_str == "Missing columns: ['c']"
    assert missing_cols == ["c"]

    res, report_str, missing_cols = util.report_missing_columns(df, ["a", "b", "c"])
    assert not res
    assert report_str == "Missing columns: ['c']"
    assert missing_cols == ["c"]


def test_check_missing_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], ("c", "d"): [5, 6]})

    assert util.check_missing_columns(df, ["a"]) == []
    assert util.check_missing_columns(df, ["c"]) == ["c"]
    assert util.check_missing_columns(df, [("c", "d")]) == []
    assert util.check_missing_columns(df, [("d", "c")]) == [("d", "c")]

    df.columns = pd.MultiIndex.from_tuples([("first_level", i) for i in df.columns.values])
    assert util.check_missing_columns(df, ["first_level"]) == []
    assert util.check_missing_columns(df, ["a"]) == ["a"]
    assert util.check_missing_columns(df, [{"first_level": ["a", "z"]}]) == [("first_level", "z")]
    assert util.check_missing_columns(df, [{"first_level": "a"}]) == []
    assert util.check_missing_columns(df, [{"unknown_first_level": "a"}]) == [
        ("unknown_first_level", "a")
    ]
    assert util.check_missing_columns(df, [{"unknown_first_level": ["a", "z"]}]) == [
        ("unknown_first_level", "a"),
        ("unknown_first_level", "z"),
    ]


def _tested_func(row, arg1, arg2):
    time.sleep(1)
    print("SOMETHING")
    print()
    if row["b"] == 6:
        raise ValueError("test 'b' value")
    return result.ValidationResult(
        True, new_data=f"{row['is_valid']}_{arg1}_{arg2}", pid=os.getpid()
    )


@pytest.mark.parametrize("nb_processes", [None, 1, 2, 100])
@pytest.mark.parametrize("redirect_stdout", [True, False])
def test_apply_to_df(nb_processes, redirect_stdout):
    df = result.ValidationResultSet(
        pd.DataFrame(
            {
                "is_valid": [True, False, True, True],
                "a": [1, 2, 3, 4],
                "b": [4, 5, 6, 7],
                ("c", "d"): [7, 8, 9, 10],
            }
        ),
        output_columns={"new_data": None, "pid": None},
    )

    res = util.apply_to_df(
        df,
        _tested_func,
        "val1",
        "val2",
        nb_processes=nb_processes,
        redirect_stdout=redirect_stdout,
    )

    res_dict = res.to_dict()
    exception = res_dict.pop("exception")
    res_dict.pop("pid")
    assert res_dict == {
        "is_valid": {0: True, 1: False, 2: False, 3: True},
        "ret_code": {0: 0, 1: 1, 2: 1, 3: 0},
        "comment": {0: None, 1: None, 2: None, 3: None},
        "a": {0: 1, 1: 2, 2: 3, 3: 4},
        "b": {0: 4, 1: 5, 2: 6, 3: 7},
        "new_data": {0: "True_val1_val2", 1: None, 2: None, 3: "True_val1_val2"},
        ("c", "d"): {0: 7, 1: 8, 2: 9, 3: 10},
    }
    assert exception[0] is None
    assert exception[1] is None
    assert exception[3] is None

    assert res["pid"].isnull().tolist() == [False, True, True, False]
    if nb_processes is None or nb_processes == 1:
        assert res.loc[0, "pid"] >= 0
        assert res.loc[0, "pid"] == res.loc[3, "pid"]
    else:
        assert res.loc[0, "pid"] >= 0
        assert res.loc[3, "pid"] >= 0
        assert res.loc[0, "pid"] != res.loc[3, "pid"]
        assert len(res.loc[~res["pid"].isnull(), "pid"].unique()) == min(nb_processes, len(df) - 2)

    exception_lines = exception[2].split("\n")
    assert exception_lines[0] == "Traceback (most recent call last):"
    assert exception_lines[2] == "    res = func(row, *args, **kwargs)"
    assert exception_lines[4] == "    raise ValueError(\"test 'b' value\")"
    assert exception_lines[5] == "ValueError: test 'b' value"
    assert (
        re.match(r"  File \"(\/.*?\.[\w:]+)\", line \d+, in try_operation", exception_lines[1])
        is not None
    )
    assert (
        re.match(r"  File \"(\/.*?\.[\w:]+)\", line \d+, in _tested_func", exception_lines[3])
        is not None
    )
