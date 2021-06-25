"""Test the data_validation_framework.result module."""
# pylint: disable=no-self-use
import pandas as pd
import pytest

from data_validation_framework import result


def test_ValidationResult():
    """Test the ValidationResult class."""
    assert result.ValidationResult(True)["ret_code"] == 0
    assert result.ValidationResult(True, 0)["ret_code"] == 0
    assert result.ValidationResult(False)["ret_code"] == 1
    assert result.ValidationResult(False, 1)["ret_code"] == 1
    assert result.ValidationResult(False, comment="test")["comment"] == "test"
    assert result.ValidationResult(False, exception="test")["exception"] == "test"
    assert result.ValidationResult(False, any_kwarg="test")["any_kwarg"] == "test"

    with pytest.raises(ValueError):
        result.ValidationResult(True, 1)

    with pytest.raises(ValueError):
        result.ValidationResult(False, 0)

    result.ValidationResult(True, 2, comment="test comment")

    with pytest.warns(
        UserWarning, match="A comment should be set when the 'ret_code' is greater than 1."
    ):
        result.ValidationResult(True, 2)


class TestValidationResultSet:
    """Test the ValidationResultSet class."""

    def test_defaults(self):
        """Check defaults."""
        df = result.ValidationResultSet(index=["a", "b"])
        assert df.to_dict() == {
            "is_valid": {"a": True, "b": True},
            "ret_code": {"a": 0, "b": 0},
            "comment": {"a": None, "b": None},
            "exception": {"a": None, "b": None},
        }

    def test_defaults_with_extra_data(self):
        """Check defaults with extra data."""
        df = result.ValidationResultSet({"a": [1, 2], "b": [3, 4]})
        assert df.to_dict() == {
            "is_valid": {0: True, 1: True},
            "ret_code": {0: 0, 1: 0},
            "comment": {0: None, 1: None},
            "exception": {0: None, 1: None},
            "a": {0: 1, 1: 2},
            "b": {0: 3, 1: 4},
        }

    def test_default_ret_code(self):
        """Check default return code."""
        df = result.ValidationResultSet({"is_valid": [True, False, True, False]})
        assert df.to_dict() == {
            "is_valid": {0: True, 1: False, 2: True, 3: False},
            "ret_code": {0: 0, 1: 1, 2: 0, 3: 1},
            "comment": {0: None, 1: None, 2: None, 3: None},
            "exception": {0: None, 1: None, 2: None, 3: None},
        }

    def test_output_columns(self):
        """Check output_columns argument."""
        df = result.ValidationResultSet(
            index=["a", "b"], output_columns={"col1": "val1", "col2": "val2"}
        )
        assert df.to_dict() == {
            "is_valid": {"a": True, "b": True},
            "ret_code": {"a": 0, "b": 0},
            "comment": {"a": None, "b": None},
            "exception": {"a": None, "b": None},
            "col1": {"a": "val1", "b": "val1"},
            "col2": {"a": "val2", "b": "val2"},
        }

    def test_output_columns_with_existing_columns(self):
        """Check output_columns argument with existing columns."""
        df = result.ValidationResultSet(
            {"is_valid": [True, False], "col1": ["test1", "test2"]},
            index=["a", "b"],
            output_columns={"col1": "val1", "col2": "val2"},
        )
        assert df.to_dict() == {
            "is_valid": {"a": True, "b": False},
            "ret_code": {"a": 0, "b": 1},
            "comment": {"a": None, "b": None},
            "exception": {"a": None, "b": None},
            "col1": {"a": "test1", "b": "test2"},
            "col2": {"a": "val2", "b": "val2"},
        }

    def test_constructor_df(self):
        """Check constructor with a pandas.DataFrame."""
        df = pd.DataFrame(
            {
                "is_valid": [True, False],
                "ret_code": [0, 1],
                "comment": [None, None],
                "exception": [None, None],
                "col1": ["test1", "test2"],
            },
            index=["a", "b"],
        )
        new_df = result.ValidationResultSet(df)
        assert new_df.equals(df)

    def test_constructor_ValidationResultSet(self):
        """Check constructor with a pandas.DataFrame."""
        df = result.ValidationResultSet(
            {"is_valid": [True, False], "col1": ["test1", "test2"]},
            index=["a", "b"],
            output_columns={"col1": "val1", "col2": "val2"},
        )
        new_df = result.ValidationResultSet(df)
        assert new_df.equals(df)

    @pytest.mark.filterwarnings("ignore::numpy.VisibleDeprecationWarning")
    def test_column_order(self):
        """Check column order."""
        df = result.ValidationResultSet(
            {"is_valid": [True, False], "col1": ["test1", "test2"]},
            index=["a", "b"],
            output_columns={"col1": "val1", "col2": "val2"},
        )
        df_order = result.ValidationResultSet(
            pd.DataFrame(df[["col2", "comment", "is_valid", "col1", "exception", "ret_code"]])
        )
        assert df_order.columns.tolist() == [
            "is_valid",
            "ret_code",
            "comment",
            "exception",
            "col1",
            "col2",
        ]

        df[("tuple", "in", "col")] = df["col1"]
        df_order = result.ValidationResultSet(
            pd.DataFrame(
                df[
                    [
                        "col2",
                        "comment",
                        "is_valid",
                        "col1",
                        ("tuple", "in", "col"),
                        "exception",
                        "ret_code",
                    ]
                ]
            )
        )
        assert df_order.columns.tolist() == [
            "is_valid",
            "ret_code",
            "comment",
            "exception",
            "col1",
            "col2",
            ("tuple", "in", "col"),
        ]

    def test_pandas_method(self):
        """Check using as a pandas.DataFrame."""
        df = result.ValidationResultSet(
            {"is_valid": [True, False], "col1": ["test1", "test2"]},
            index=["a", "b"],
            output_columns={"col1": "val1", "col2": "val2"},
        )
        new_df = result.ValidationResultSet(df)
        assert new_df.apply(lambda x: x.ret_code * 5, axis=1).to_dict() == {"a": 0, "b": 5}
        assert new_df["is_valid"].sum() == 1
        assert new_df["ret_code"].sum() == 1
