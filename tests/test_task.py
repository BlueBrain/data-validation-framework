"""Test the data_validation_framework.task module."""
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=no-self-use
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-lines
# pylint: disable=unused-argument
import json
import logging
import re
import time
from shutil import which

import luigi
import luigi_tools
import numpy as np
import pandas as pd
import pause
import pytest
from diff_pdf_visually import pdfdiff
from luigi_tools.parameter import OptionalStrParameter

from data_validation_framework import report
from data_validation_framework import result
from data_validation_framework import target
from data_validation_framework import task

SKIP_IF_NO_LATEXMK = not which("latexmk")
REASON_NO_LATEXMK = "The command latexmk is not available."


@pytest.fixture
def dataset_df_path(tmpdir):
    dataset_df_path = tmpdir / "dataset.csv"
    base_dataset_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    base_dataset_df.to_csv(dataset_df_path)

    return str(dataset_df_path)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_TagResultOutputMixin(tmpdir):
    """Test the data_validation_framework.task.TagResultOutputMixin class."""

    class TestTask(task.TagResultOutputMixin, luigi.Task):

        with_conflict = luigi.BoolParameter(default=False)

        def run(self):
            if not self.tag_output:
                assert self.output().path == str(tmpdir / "out" / "file.test")
            elif not self.with_conflict:
                assert re.match(
                    f"{tmpdir}/out" + r"_\d{8}-\d{2}h\d{2}m\d{2}s/file.test", self.output().path
                )
            else:
                assert re.match(
                    f"{tmpdir}/out" + r"_\d{8}-\d{2}h\d{2}m\d{2}s_\d+/file.test", self.output().path
                )

        def output(self):
            return target.TaggedOutputLocalTarget("file.test")

    root = tmpdir / "out"
    assert luigi.build([TestTask(result_path=str(root))], local_scheduler=True)

    # Test tag name conflicts
    pause.until(int(time.time()) + 1.01)
    assert luigi.build(
        [TestTask(result_path=str(root), tag_output=True, with_conflict=False)],
        local_scheduler=True,
    )
    assert luigi.build(
        [TestTask(result_path=str(root), tag_output=True, with_conflict=True)],
        local_scheduler=True,
    )


class TestSetValidationTask:
    """Test the data_validation_framework.task.SetValidationTask class."""

    @pytest.fixture
    def TestTask(self, tmpdir):

        target.TaggedOutputLocalTarget.set_default_prefix(tmpdir)

        class TestTask(task.SetValidationTask):
            @staticmethod
            def validation_function(df, output_path, *args, **kwargs):
                assert df["is_valid"].all()
                assert (df["ret_code"] == 0).all()
                assert df["comment"].isnull().all()
                assert df["exception"].isnull().all()
                assert df["a"].tolist() == [1, 2]
                assert df["b"].tolist() == [3, 4]
                df["a"] *= 10
                df[["a", "b"]].to_csv(output_path / "test.csv")

        return TestTask

    def test_defaults(self, TestTask, dataset_df_path, tmpdir):
        # Test defaults
        assert luigi.build(
            [TestTask(dataset_df=dataset_df_path, result_path=str(tmpdir / "out_defaults"))],
            local_scheduler=True,
        )

        result = pd.read_csv(tmpdir / "out_defaults" / "TestTask" / "data" / "test.csv")
        expected = pd.read_csv(tmpdir / "dataset.csv")
        expected["a"] *= 10
        assert result.equals(expected)

    def test_no_dataset_no_input(self, TestTask, tmpdir):
        # Test with no dataset and no input (should fail)

        class FailingTestTask(TestTask):
            pass

        failed_tasks = []
        exceptions = []

        @FailingTestTask.event_handler(luigi.Event.FAILURE)
        def check_exception(failed_task, exception):  # pylint: disable=unused-variable
            failed_tasks.append(str(failed_task))
            exceptions.append(str(exception))

        assert not luigi.build(
            [FailingTestTask(result_path=str(tmpdir / "out_fail"))], local_scheduler=True
        )
        assert failed_tasks == [str(FailingTestTask(result_path=str(tmpdir / "out_fail")))]
        assert exceptions == [
            str(ValueError("Either the 'dataset_df' parameter or a requirement must be provided."))
        ]

    def test_inputs_outputs(self, TestTask, dataset_df_path, tmpdir):
        # Test inputs

        class TestTaskWithOutputs(task.SetValidationTask):

            output_columns = {"a": None}

            @staticmethod
            def validation_function(df, output_path, *args, **kwargs):
                assert df["is_valid"].all()
                assert (df["ret_code"] == 0).all()
                assert df["comment"].isnull().all()
                assert df["exception"].isnull().all()
                assert df["a"].tolist() == [1, 2]
                assert df["b"].tolist() == [3, 4]
                df["a"] *= 10
                df[["a", "b"]].to_csv(output_path / "test.csv")

        class TestTaskWithInputs(task.SetValidationTask):
            def inputs(self):
                return {
                    TestTaskWithOutputs(dataset_df=dataset_df_path, result_path=self.result_path): {
                        "a": "a_input"
                    }
                }

            def kwargs(self):
                return {"dataset_df": self.dataset_df}

            @staticmethod
            def validation_function(df, output_path, *args, **kwargs):
                assert df["is_valid"].all()
                assert (df["ret_code"] == 0).all()
                assert df["comment"].isnull().all()
                assert df["exception"].isnull().all()
                assert df["a_input"].tolist() == [10, 20]

                if kwargs["dataset_df"] is not None:
                    assert df["a"].tolist() == [1, 2]
                    assert df["b"].tolist() == [3, 4]

                df["a_input"] *= 10
                df.to_csv(output_path / "test.csv")

        # Test with inputs but no dataset
        assert luigi.build(
            [TestTaskWithInputs(result_path=str(tmpdir / "out_inputs"))], local_scheduler=True
        )

        result = pd.read_csv(tmpdir / "out_inputs" / "TestTaskWithInputs" / "data" / "test.csv")
        expected = pd.read_csv(tmpdir / "dataset.csv")
        expected["a_input"] = expected["a"] * 100
        assert (result["a_input"] == expected["a_input"]).all()

        # Test with inputs and dataset
        assert luigi.build(
            [
                TestTaskWithInputs(
                    dataset_df=dataset_df_path, result_path=str(tmpdir / "out_inputs_and_outputs")
                )
            ],
            local_scheduler=True,
        )

        result = pd.read_csv(
            tmpdir / "out_inputs_and_outputs" / "TestTaskWithInputs" / "data" / "test.csv"
        )
        expected = pd.read_csv(tmpdir / "dataset.csv")
        expected["a_input"] = expected["a"] * 100
        assert result[["a", "b", "a_input"]].equals(expected[["a", "b", "a_input"]])

    def test_missing_columns(self, TestTask, dataset_df_path, tmpdir):
        # Test missing columns in requirements

        class TestTaskMissingColumns(task.SetValidationTask):
            def inputs(self):
                return {TestTask(): {"a": "a_input"}}

            @staticmethod
            def validation_function(df, output_path, *args, **kwargs):
                assert df["is_valid"].all()
                assert (df["ret_code"] == 0).all()
                assert df["comment"].isnull().all()
                assert df["exception"].isnull().all()
                assert df["a_input"].tolist() == [10, 20]

                df["a_input"] *= 10
                df.to_csv(output_path / "test.csv")

        failed_tasks = []
        exceptions = []

        @TestTaskMissingColumns.event_handler(luigi.Event.FAILURE)
        def check_exception(failed_task, exception):  # pylint: disable=unused-variable
            failed_tasks.append(str(failed_task))
            exceptions.append(str(exception))

        assert not luigi.build(
            [
                TestTaskMissingColumns(
                    dataset_df=dataset_df_path, result_path=str(tmpdir / "out_missing_columns")
                )
            ],
            local_scheduler=True,
        )

        assert failed_tasks == [
            str(
                TestTaskMissingColumns(
                    dataset_df=dataset_df_path, result_path=str(tmpdir / "out_missing_columns")
                )
            )
        ]
        assert exceptions == [
            str(
                KeyError(
                    "The columns ['a'] are missing from the output columns of "
                    "TestTask("
                    "tag_output=False, result_path=, dataset_df=, input_index_col=, data_dir=data"
                    ").",
                )
            )
        ]

    class TestPropagation:
        @pytest.fixture
        def BaseTestTask(self, TestTask):
            class BaseTestTask(TestTask):
                def kwargs(self):
                    return {
                        "dataset_df": str(self.dataset_df),
                        "result_path": str(self.result_path),
                    }

                @staticmethod
                def validation_function(df, output_path, *args, **kwargs):
                    with open(output_path / "test.json", "w") as f:
                        json.dump(kwargs, f)

                    TestTask.validation_function(df, output_path, *args, **kwargs)

            return BaseTestTask

        @pytest.fixture
        def TestTaskPassDatasetAndResultPath(self, BaseTestTask):
            class TestTaskPassDatasetAndResultPath(task.SetValidationTask):
                def inputs(self):
                    return {BaseTestTask(): {}}

                @staticmethod
                def validation_function(df, output_path, *args, **kwargs):
                    assert df["is_valid"].all()
                    assert (df["ret_code"] == 0).all()
                    assert df["comment"].isnull().all()
                    assert df["exception"].isnull().all()
                    assert df["a"].tolist() == [1, 2]

                    df["a"] *= 100
                    df.to_csv(output_path / "test.csv")

            return TestTaskPassDatasetAndResultPath

        def test_dataset_propagation(
            self, TestTaskPassDatasetAndResultPath, dataset_df_path, tmpdir
        ):
            # Test that the dataset is properly passed to the requirements
            assert luigi.build(
                [
                    TestTaskPassDatasetAndResultPath(
                        dataset_df=dataset_df_path, result_path=str(tmpdir / "out_pass_dataset")
                    )
                ],
                local_scheduler=True,
            )

            with open(tmpdir / "out_pass_dataset" / "BaseTestTask" / "data" / "test.json") as f:
                params = json.load(f)

            assert params == {
                "dataset_df": f"{tmpdir}/dataset.csv",
                "result_path": f"{tmpdir}/out_pass_dataset",
            }

            result_1 = pd.read_csv(
                tmpdir / "out_pass_dataset" / "BaseTestTask" / "data" / "test.csv"
            )
            result_2 = pd.read_csv(
                tmpdir
                / "out_pass_dataset"
                / "TestTaskPassDatasetAndResultPath"
                / "data"
                / "test.csv"
            )
            expected = pd.read_csv(tmpdir / "dataset.csv")
            expected["a"] *= 10
            assert result_1.equals(expected)
            expected["a"] *= 10
            assert result_2[["a", "b"]].equals(expected[["a", "b"]])

        def test_dataset_propagation_with_config(
            self, TestTaskPassDatasetAndResultPath, dataset_df_path, tmpdir
        ):
            # Test that the dataset is not propagated when a value is already given in the config
            with luigi_tools.util.set_luigi_config(
                {
                    "BaseTestTask": {
                        "result_path": str(tmpdir / "specific_out_path"),
                    }
                }
            ):
                assert luigi.build(
                    [
                        TestTaskPassDatasetAndResultPath(
                            dataset_df=dataset_df_path, result_path=str(tmpdir / "out_pass_dataset")
                        )
                    ],
                    local_scheduler=True,
                )

            with open(tmpdir / "specific_out_path" / "BaseTestTask" / "data" / "test.json") as f:
                params = json.load(f)

            assert params == {
                "dataset_df": f"{tmpdir}/dataset.csv",
                "result_path": f"{tmpdir}/specific_out_path",
            }

            result_1 = pd.read_csv(
                tmpdir / "specific_out_path" / "BaseTestTask" / "data" / "test.csv"
            )
            result_2 = pd.read_csv(
                tmpdir
                / "out_pass_dataset"
                / "TestTaskPassDatasetAndResultPath"
                / "data"
                / "test.csv"
            )
            expected = pd.read_csv(tmpdir / "dataset.csv")
            expected["a"] *= 10
            assert result_1.equals(expected)
            expected["a"] *= 10
            assert result_2[["a", "b"]].equals(expected[["a", "b"]])

    def test_failing_validation_function(self, TestTask, dataset_df_path, tmpdir):
        # Test with a failing validation function

        class FailingTestTask(TestTask):
            @staticmethod
            def validation_function(*args, **kwargs):
                raise RuntimeError("This function always fails")

        assert luigi.build(
            [
                FailingTestTask(
                    dataset_df=dataset_df_path,
                    result_path=str(tmpdir / "out_failing_validation_function"),
                )
            ],
            local_scheduler=True,
        )

        result = pd.read_csv(
            tmpdir / "out_failing_validation_function" / "FailingTestTask" / "report.csv"
        )
        assert (~result["is_valid"]).all()
        assert (result["ret_code"] == 1).all()
        assert (result["comment"].isnull()).all()
        assert (
            result["exception"].str.split("\n").apply(lambda x: x[5])
            == "RuntimeError: This function always fails"
        ).all()

    def test_task_name(self, dataset_df_path, tmpdir):
        # Test with a custom task name

        class TestTask(task.SetValidationTask):
            task_name = "test_custom_task_name"

            @staticmethod
            def validation_function(*args, **kwargs):
                pass

        assert luigi.build(
            [
                TestTask(
                    dataset_df=dataset_df_path,
                    result_path=str(tmpdir),
                )
            ],
            local_scheduler=True,
        )
        assert (tmpdir / "test_custom_task_name" / "report.csv").exists()

    def test_duplicated_index(self, tmpdir, TestTask):
        dataset_df_path = str(tmpdir / "dataset.csv")
        base_dataset_df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}, index=[0, 1, 1, 0])
        base_dataset_df.to_csv(dataset_df_path, index=True, index_label="index_col")

        failed_tasks = []
        exceptions = []

        @TestTask.event_handler(luigi.Event.FAILURE)
        def check_exception(failed_task, exception):  # pylint: disable=unused-variable
            failed_tasks.append(str(failed_task))
            exceptions.append(str(exception))

        failing_task = TestTask(
            dataset_df=dataset_df_path,
            input_index_col="index_col",
            result_path=str(tmpdir / "out_failing_duplicated_index"),
        )
        assert not luigi.build([failing_task], local_scheduler=True)

        assert failed_tasks == [str(failing_task)]
        assert exceptions == [str(IndexError("The following index values are duplicated: [0, 1]"))]

    def test_change_index(self, tmpdir, TestTask):
        dataset_df_path = str(tmpdir / "dataset.csv")
        base_dataset_df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}, index=[0, 1, 2, 3])
        base_dataset_df.to_csv(dataset_df_path, index=True, index_label="index_col")

        class TestTaskUpdateIndex(TestTask):
            @staticmethod
            def validation_function(df, output_path, *args, **kwargs):
                df.sort_index(ascending=False, inplace=True)

        failed_tasks = []
        exceptions = []

        @TestTaskUpdateIndex.event_handler(luigi.Event.FAILURE)
        def check_exception(failed_task, exception):  # pylint: disable=unused-variable
            failed_tasks.append(str(failed_task))
            exceptions.append(str(exception))

        failing_task = TestTaskUpdateIndex(
            dataset_df=dataset_df_path,
            input_index_col="index_col",
            result_path=str(tmpdir / "out_failing_update_index"),
        )
        assert not luigi.build([failing_task], local_scheduler=True)

        assert failed_tasks == [str(failing_task)]
        assert exceptions == [
            str(
                IndexError(
                    "The index changed during the process. Please update your validation function "
                    "or your pre/post process functions to avoid this behaviour."
                )
            )
        ]

    def test_missing_retcodes(self, tmpdir, dataset_df_path, TestTask):
        class TestTaskMissingRetcodes(TestTask):
            mode = OptionalStrParameter(default=None)

            def kwargs(self):
                return {"mode": self.mode}

            @staticmethod
            def validation_function(df, output_path, *args, **kwargs):
                if kwargs["mode"] == "valid":
                    df["is_valid"] = True
                    df["ret_code"] = 1
                elif kwargs["mode"] == "not valid":
                    df["is_valid"] = False
                    df["ret_code"] = 0

        failed_tasks = []
        exceptions = []

        @TestTaskMissingRetcodes.event_handler(luigi.Event.FAILURE)
        def check_exception(failed_task, exception):  # pylint: disable=unused-variable
            failed_tasks.append(str(failed_task))
            exceptions.append(str(exception))

        failing_task_valid = TestTaskMissingRetcodes(
            dataset_df=dataset_df_path,
            result_path=str(tmpdir / "out_failing_update_index"),
            mode="valid",
        )
        failing_task_notvalid = TestTaskMissingRetcodes(
            dataset_df=dataset_df_path,
            result_path=str(tmpdir / "out_failing_update_index"),
            mode="not valid",
        )
        assert not luigi.build([failing_task_valid], local_scheduler=True)
        assert not luigi.build([failing_task_notvalid], local_scheduler=True)

        assert failed_tasks == [str(failing_task_valid), str(failing_task_notvalid)]
        assert (
            exceptions
            == [
                str(
                    task.ValidationError(
                        "The 'ret_code' values are not consistent with the 'is_valid' values."
                    )
                )
            ]
            * 2
        )

    def test_rename_multiindex(self):
        df_1_level = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}, index=[0, 1, 2, 3])

        task.BaseValidationTask._rename_cols(df_1_level)  # pylint: disable=protected-access
        assert df_1_level.columns.tolist() == ["a", "b"]

        df_2_levels = pd.DataFrame(
            {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "Unnamed: c": [9, 10, 11, 12]},
            index=[0, 1, 2, 3],
        )
        df_2_levels.columns = pd.MultiIndex.from_tuples(
            [("level_1", i) for i in df_2_levels.columns.values]
        )
        task.BaseValidationTask._rename_cols(df_2_levels)  # pylint: disable=protected-access
        assert df_2_levels.columns.tolist() == [("level_1", "a"), ("level_1", "b"), ("level_1", "")]

    def test_check_inputs(self, TestTask):
        assert not task.BaseValidationTask.check_inputs({})
        assert not task.BaseValidationTask.check_inputs(None)

        with pytest.raises(
            ValueError,
            match=(
                r"The destination column of the task TestTask\(tag_output=False, result_path=, "
                r"dataset_df=, input_index_col=, data_dir=data\) can not be one of "
                r"\['is_valid', 'ret_code', 'comment', 'exception'\]\."
            ),
        ):
            task.BaseValidationTask.check_inputs(
                {
                    TestTask(): {"a": "is_valid"},
                }
            )

        class TestConflictingColumns(task.SetValidationTask):
            pass

        with pytest.raises(
            ValueError,
            match=(
                r"The destination column 'a_conflict' of the tasks TestConflictingColumns\("
                r"tag_output=False, result_path=, dataset_df=, input_index_col=, data_dir=data\) "
                r"and TestTask\(tag_output=False, result_path=, dataset_df=, input_index_col=, "
                r"data_dir=data\) are conflicting\."
            ),
        ):
            task.BaseValidationTask.check_inputs(
                {TestTask(): {"a": "a_conflict"}, TestConflictingColumns(): {"a": "a_conflict"}}
            )

    def test_static_args_kwargs(self, dataset_df_path):
        class TestTask(task.ElementValidationTask):

            args = [1, "a"]
            kwargs = {"int_value": 1, "str_value": "a"}

            @staticmethod
            def validation_function(df, output_path, *args, **kwargs):
                assert args == [1, "a"]
                assert kwargs == {"int_value": 1, "str_value": "a"}

        assert luigi.build(
            [TestTask(dataset_df=dataset_df_path)],
            local_scheduler=True,
        )

        class TestFailingArgsTask(task.ElementValidationTask):

            args = 1

            @staticmethod
            def validation_function(df, output_path, *args, **kwargs):
                assert args == 1

        failed_tasks = []
        exceptions = []

        @TestFailingArgsTask.event_handler(luigi.Event.FAILURE)
        def check_exception_args(failed_task, exception):  # pylint: disable=unused-variable
            failed_tasks.append(str(failed_task))
            exceptions.append(str(exception))

        assert not luigi.build(
            [TestFailingArgsTask(dataset_df=dataset_df_path)],
            local_scheduler=True,
        )

        assert failed_tasks == [str(TestFailingArgsTask(dataset_df=dataset_df_path))]
        assert exceptions == [
            str(
                TypeError(
                    "The 'args' must either be a method returning a list or a tuple, or an actual "
                    "list or tuple."
                )
            )
        ]

        class TestFailingKwargsTask(task.ElementValidationTask):

            kwargs = 1

            @staticmethod
            def validation_function(df, output_path, *args, **kwargs):
                assert kwargs == 1

        failed_tasks = []
        exceptions = []

        @TestFailingKwargsTask.event_handler(luigi.Event.FAILURE)
        def check_exception_kwargs(failed_task, exception):  # pylint: disable=unused-variable
            failed_tasks.append(str(failed_task))
            exceptions.append(str(exception))

        assert not luigi.build(
            [TestFailingKwargsTask(dataset_df=dataset_df_path)],
            local_scheduler=True,
        )

        assert failed_tasks == [str(TestFailingKwargsTask(dataset_df=dataset_df_path))]
        assert exceptions == [
            str(
                TypeError(
                    "The 'kwargs' must either be a method returning a dict, or an actual dict."
                )
            )
        ]

    def test_different_input_index(self, tmpdir, dataset_df_path, caplog):
        class TestTask(task.ElementValidationTask):
            @staticmethod
            def validation_function(df, output_path, *args, **kwargs):
                pass

        class TestDifferentInputIndex(task.ElementValidationTask):
            def inputs(self):
                return {
                    TestTask(
                        dataset_df=dataset_df_path,
                        input_index_col=self.input_index_col,
                    ): {}
                }

            @staticmethod
            def validation_function(df, output_path, *args, **kwargs):
                pass

        updated_dataset = pd.read_csv(dataset_df_path)
        updated_dataset.index += 1
        name_parts = dataset_df_path.rsplit(".", 1)
        name_parts[0] += "_updated"
        new_df_path = ".".join(name_parts)
        updated_dataset.to_csv(new_df_path)
        caplog.clear()
        caplog.set_level(logging.DEBUG)
        assert luigi.build(
            [
                TestDifferentInputIndex(
                    dataset_df=new_df_path,
                    result_path=str(tmpdir / "different_input_index"),
                    input_index_col=0,
                )
            ],
            local_scheduler=True,
        )
        res = [
            i
            for i in caplog.record_tuples
            if i[0] == "data_validation_framework.task" and i[1] == logging.WARNING
        ]
        assert len(res) == 1
        assert res[0][2] == (
            "The following inconsistent indexes between the dataset and the inputs are "
            "ignored: [2]"
        )

    def test_external_function(self, dataset_df_path):
        def external_function(df, output_path, *args, **kwargs):
            assert args == [1, "a"]
            assert kwargs == {"k1": 1, "k2": 2}

        class TestExternalFunctionTask(task.ElementValidationTask):

            args = [1, "a"]
            kwargs = {"k1": 1, "k2": 2}
            validation_function = external_function

        assert luigi.build(
            [TestExternalFunctionTask(dataset_df=dataset_df_path)],
            local_scheduler=True,
        )


class TestElementValidationTask:
    """Test the data_validation_framework.task.ElementValidationTask class."""

    @pytest.fixture
    def TestTask(self, tmpdir):

        target.TaggedOutputLocalTarget.set_default_prefix(tmpdir)

        class TestTask(task.ElementValidationTask):
            @staticmethod
            # pylint: disable=arguments-differ
            def validation_function(row, output_path, *args, **kwargs):
                if row["a"] <= 1:
                    return result.ValidationResult(is_valid=True)
                if row["a"] <= 2:
                    return result.ValidationResult(is_valid=False, comment="bad value")
                raise ValueError(f"Incorrect value {row['a']}")

        return TestTask

    @pytest.fixture
    def dataset_df_path(self, tmpdir):
        dataset_df_path = tmpdir / "dataset.csv"
        base_dataset_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        base_dataset_df.to_csv(dataset_df_path)

        return str(dataset_df_path)

    def test_defaults(self, TestTask, dataset_df_path, tmpdir):
        # Test that the dataset is properly passed to the requirements

        assert luigi.build(
            [TestTask(dataset_df=dataset_df_path, result_path=str(tmpdir / "out"))],
            local_scheduler=True,
        )
        result = pd.read_csv(tmpdir / "out" / "TestTask" / "report.csv")
        assert result["is_valid"].tolist() == [True, False, False]
        assert result["ret_code"].tolist() == [0, 1, 1]
        assert result["comment"].tolist() == [np.nan, "bad value", np.nan]
        assert result.loc[[0, 1], "exception"].isnull().all()
        assert result.loc[2, "exception"].split("\n")[5] == "ValueError: Incorrect value 3"

    @pytest.mark.parametrize("nb_processes", [None, 1, 5])
    def test_nb_processes(self, TestTask, dataset_df_path, tmpdir, nb_processes):
        # Test that the number of processes is properly passed to the requirements

        class TestWorkflow(task.ValidationWorkflow):
            def inputs(self):
                return {TestTask(): {}}

        assert (
            TestWorkflow(
                dataset_df=dataset_df_path,
                result_path=str(tmpdir / "out"),
                nb_processes=nb_processes,
            )
            .requires()[0]
            .nb_processes
            == nb_processes
        )


class TestValidationWorkflow:
    """Test the data_validation_framework.task.ValidationWorkflow class."""

    @pytest.fixture
    def comment(self):
        return "This element is not valid"

    @pytest.fixture
    def exception(self):
        return (
            "Traceback (most recent call last):"
            '  File "test.py", line 1, in <module>'
            '    raise ValueError("This element is not valid")'
            "ValueError: This element is not valid"
        )

    @pytest.fixture
    def default_report_config_test_date(self):
        report._DEFAULT_REPORT_CONFIG["today"] = "TEST DATE"  # pylint: disable=protected-access
        yield None
        report._DEFAULT_REPORT_CONFIG.pop("today")  # pylint: disable=protected-access

    def test_gather(self, tmpdir, dataset_df_path, comment, exception):
        class TestTask(task.SetValidationTask):
            @staticmethod
            def validation_function(df, output_path, *args, **kwargs):
                df.loc[1, "is_valid"] = False
                df.loc[1, "ret_code"] = 1
                df.loc[1, "comment"] = comment
                df.loc[1, "exception"] = exception

        class TestWorkflow(task.ValidationWorkflow):
            def inputs(self):
                return {TestTask(): {}}

            @staticmethod
            def validation_function(df, output_path, *args, **kwargs):
                df.to_csv(output_path / "test_gather.csv")

        assert luigi.build(
            [TestWorkflow(dataset_df=dataset_df_path, result_path=str(tmpdir))],
            local_scheduler=True,
        )

        assert (tmpdir / "dataset.csv").exists()
        assert (tmpdir / "TestTask" / "report.csv").exists()
        assert (tmpdir / "TestWorkflow" / "report.csv").exists()

        result = pd.read_csv(tmpdir / "TestWorkflow" / "report.csv")
        expected = pd.DataFrame(
            {
                "__index_label__": [0, 1],
                "is_valid": [True, False],
                "ret_code": [0, 1],
                "('TestTask', 'is_valid')": [True, False],
                "('TestTask', 'ret_code')": [0, 1],
            }
        )
        assert result.columns.tolist() == [
            "__index_label__",
            "is_valid",
            "ret_code",
            "comment",
            "exception",
            "('TestTask', 'is_valid')",
            "('TestTask', 'ret_code')",
            "('TestTask', 'comment')",
            "('TestTask', 'exception')",
        ]
        assert result[
            [
                "__index_label__",
                "is_valid",
                "ret_code",
                "('TestTask', 'is_valid')",
                "('TestTask', 'ret_code')",
            ]
        ].equals(expected)
        assert (
            result.loc[
                0, ["comment", "exception", "('TestTask', 'comment')", "('TestTask', 'exception')"]
            ]
            .isnull()
            .all()
            .all()
        )
        assert (result.loc[1, ["comment", "exception"]].isnull()).all()
        assert result.loc[1, "('TestTask', 'comment')"] == comment
        assert result.loc[1, "('TestTask', 'exception')"] == exception

    def test_no_report(self, tmpdir, dataset_df_path):
        class TestTask(task.SetValidationTask):
            @staticmethod
            def validation_function(*args, **kwargs):
                pass

        class TestWorkflow(task.ValidationWorkflow):
            generate_report = False

            def inputs(self):
                return {TestTask(): {}}

        assert luigi.build(
            [TestWorkflow(dataset_df=dataset_df_path, result_path=str(tmpdir))],
            local_scheduler=True,
        )
        assert (tmpdir / "TestWorkflow" / "report.csv").exists()
        assert not (tmpdir / "report.pdf").exists()

    class TestReport:
        """Test the report generation after workflow run."""

        @pytest.fixture
        def TestTask(self, comment, exception):
            class TestTask(task.SetValidationTask):
                """A test validation task."""

                no_exception = luigi.BoolParameter(default=False)
                mode = OptionalStrParameter(default=None)

                def kwargs(self):
                    return {
                        "no_exception": self.no_exception,
                        "mode": self.mode,
                    }

                @staticmethod
                def validation_function(df, output_path, *args, **kwargs):
                    if kwargs["mode"] == "all_succeed":
                        df["is_valid"] = True
                        df["ret_code"] = 0
                    elif kwargs["mode"] == "all_fail":
                        df["is_valid"] = False
                        df["ret_code"] = 1
                    else:
                        df.loc[1, "is_valid"] = False
                        df.loc[1, "ret_code"] = 1
                        if not kwargs["no_exception"]:
                            df.loc[1, "comment"] = comment
                            df.loc[1, "exception"] = exception

            return TestTask

        @pytest.fixture
        def TestTask_Specifications(self):
            class TestTask_Specifications(task.SetValidationTask):
                """A test validation task with a specific report doc."""

                __specifications__ = "The specific doc only used in report."

                @staticmethod
                def validation_function(*args, **kwargs):
                    pass

            return TestTask_Specifications

        @pytest.fixture
        def TestWorkflow(self, TestTask, TestTask_Specifications):
            class TestWorkflow(task.ValidationWorkflow):
                """The global validation workflow."""

                no_exception = luigi.BoolParameter(default=False)
                mode = OptionalStrParameter(default=None)

                def inputs(self):
                    return {
                        TestTask(no_exception=self.no_exception, mode=self.mode): {},
                        TestTask_Specifications(): {},
                    }

            return TestWorkflow

        def test_specifications(self, TestTask, TestTask_Specifications):
            assert TestTask(dataset_df="", result_path="").__specifications__ == (
                "A test validation task."
            )
            assert TestTask_Specifications(dataset_df="", result_path="").__specifications__ == (
                "The specific doc only used in report."
            )

        def test_rst2pdf(
            self, tmpdir, dataset_df_path, data_dir, TestWorkflow, default_report_config_test_date
        ):
            root = tmpdir / "rst2pdf"
            assert luigi.build(
                [TestWorkflow(dataset_df=dataset_df_path, result_path=str(root))],
                local_scheduler=True,
            )
            assert (root / "TestWorkflow" / "report.csv").exists()
            assert (root / "report_TestWorkflow.pdf").exists()
            assert pdfdiff(
                root / "report_TestWorkflow.pdf", data_dir / "test_report" / "report_rst2pdf.pdf"
            )

        @pytest.mark.skipif(SKIP_IF_NO_LATEXMK, reason=REASON_NO_LATEXMK)
        def test_latexpdf(
            self, tmpdir, dataset_df_path, data_dir, TestWorkflow, default_report_config_test_date
        ):
            root = tmpdir / "latexpdf"
            assert luigi.build(
                [
                    TestWorkflow(
                        dataset_df=dataset_df_path, result_path=str(root), report_type="latexpdf"
                    )
                ],
                local_scheduler=True,
            )
            assert (root / "TestWorkflow" / "report.csv").exists()
            assert (root / "report_TestWorkflow.pdf").exists()
            assert pdfdiff(
                root / "report_TestWorkflow.pdf", data_dir / "test_report" / "report_latexpdf.pdf"
            )

        def test_fail_element_no_exception(
            self, tmpdir, dataset_df_path, data_dir, TestWorkflow, default_report_config_test_date
        ):
            root = tmpdir / "rst2pdf_no_exception"
            assert luigi.build(
                [
                    TestWorkflow(
                        dataset_df=dataset_df_path, result_path=str(root), no_exception=True
                    )
                ],
                local_scheduler=True,
            )
            assert (root / "TestWorkflow" / "report.csv").exists()
            assert (root / "report_TestWorkflow.pdf").exists()
            assert pdfdiff(
                root / "report_TestWorkflow.pdf",
                data_dir / "test_report" / "report_no_exception_rst2pdf.pdf",
            )

        def test_report_relative_path(
            self, tmpdir, dataset_df_path, data_dir, TestWorkflow, default_report_config_test_date
        ):
            root = tmpdir / "relative_path"
            assert luigi.build(
                [
                    TestWorkflow(
                        dataset_df=dataset_df_path,
                        result_path=str(root),
                        report_path="report_test_name.pdf",
                    )
                ],
                local_scheduler=True,
            )
            assert (root / "TestWorkflow" / "report.csv").exists()
            assert (root / "report_test_name.pdf").exists()
            assert pdfdiff(
                root / "report_test_name.pdf", data_dir / "test_report" / "report_rst2pdf.pdf"
            )

        def test_report_absolute_path(
            self, tmpdir, dataset_df_path, data_dir, TestWorkflow, default_report_config_test_date
        ):
            root = tmpdir / "absolute_path"
            assert luigi.build(
                [
                    TestWorkflow(
                        dataset_df=dataset_df_path,
                        result_path=str(root),
                        report_path=str(tmpdir / "report_test_absolute.pdf"),
                    )
                ],
                local_scheduler=True,
            )
            assert (root / "TestWorkflow" / "report.csv").exists()
            assert (tmpdir / "report_test_absolute.pdf").exists()
            assert pdfdiff(
                tmpdir / "report_test_absolute.pdf", data_dir / "test_report" / "report_rst2pdf.pdf"
            )

        def test_report_all_succeed(
            self, tmpdir, dataset_df_path, data_dir, TestWorkflow, default_report_config_test_date
        ):
            root = tmpdir / "absolute_path"
            assert luigi.build(
                [
                    TestWorkflow(
                        dataset_df=dataset_df_path,
                        result_path=str(root),
                        report_path=str(tmpdir / "report_test_all_success.pdf"),
                        mode="all_succeed",
                    )
                ],
                local_scheduler=True,
            )
            assert (root / "TestWorkflow" / "report.csv").exists()
            assert (tmpdir / "report_test_all_success.pdf").exists()
            assert pdfdiff(
                tmpdir / "report_test_all_success.pdf",
                data_dir / "test_report" / "report_rst2pdf_all_success.pdf",
            )

        def test_report_all_fail(
            self, tmpdir, dataset_df_path, data_dir, TestWorkflow, default_report_config_test_date
        ):
            root = tmpdir / "absolute_path"
            assert luigi.build(
                [
                    TestWorkflow(
                        dataset_df=dataset_df_path,
                        result_path=str(root),
                        report_path=str(tmpdir / "report_test_all_fail.pdf"),
                        mode="all_fail",
                    )
                ],
                local_scheduler=True,
            )
            assert (root / "TestWorkflow" / "report.csv").exists()
            assert (tmpdir / "report_test_all_fail.pdf").exists()
            assert pdfdiff(
                tmpdir / "report_test_all_fail.pdf",
                data_dir / "test_report" / "report_rst2pdf_all_fail.pdf",
            )

    class TestReportBeforeRun:
        """Test the report generation before workflow run (generate only the specifications)."""

        @pytest.fixture
        def TestTask(self):
            class TestTask(task.SetValidationTask):
                """A test validation task."""

                def run(self):
                    raise RuntimeError("THIS TASK SHOULD NOT BE RUN")

            return TestTask

        @pytest.fixture
        def TestTask_Specifications(self):
            class TestTask_Specifications(task.SetValidationTask):
                """A test validation task with a specific report doc."""

                __specifications__ = "The specific doc only used in report."

                def run(self):
                    raise RuntimeError("THIS TASK SHOULD NOT BE RUN")

            return TestTask_Specifications

        @pytest.fixture
        def TestWorkflow(self, TestTask, TestTask_Specifications):
            class TestWorkflow(task.ValidationWorkflow):
                """The global validation workflow."""

                def inputs(self):
                    return {
                        TestTask(): {},
                        TestTask_Specifications(): {},
                    }

            return TestWorkflow

        @pytest.fixture
        def TestNestedWorkflow(self, TestTask, TestWorkflow):
            class TestNestedWorkflow(task.ValidationWorkflow):
                """The global validation workflow."""

                def inputs(self):
                    return {
                        TestTask(): {},
                        TestWorkflow(): {},
                    }

            return TestNestedWorkflow

        def test_rst2pdf(self, tmpdir, dataset_df_path, data_dir, TestWorkflow):
            root = tmpdir / "rst2pdf"
            assert luigi.build(
                [
                    TestWorkflow(
                        dataset_df=dataset_df_path,
                        result_path=str(root),
                        specifications_only=True,
                    )
                ],
                local_scheduler=True,
            )

            assert (root / "TestWorkflow_specifications.pdf").exists()
            assert pdfdiff(
                root / "TestWorkflow_specifications.pdf",
                data_dir / "test_report_before_run" / "report_rst2pdf.pdf",
                threshold=25,
            )

        def test_rst2pdf_report_path(self, tmpdir, dataset_df_path, data_dir, TestWorkflow):
            root = tmpdir / "rst2pdf"
            assert luigi.build(
                [
                    TestWorkflow(
                        dataset_df=dataset_df_path,
                        result_path=str(root),
                        report_path="test_custom_document_name.pdf",
                        specifications_only=True,
                    )
                ],
                local_scheduler=True,
            )

            assert (root / "test_custom_document_name.pdf").exists()
            assert pdfdiff(
                root / "test_custom_document_name.pdf",
                data_dir / "test_report_before_run" / "report_rst2pdf.pdf",
                threshold=25,
            )

        @pytest.mark.skipif(SKIP_IF_NO_LATEXMK, reason=REASON_NO_LATEXMK)
        def test_latexpdf(self, tmpdir, dataset_df_path, data_dir, TestWorkflow):
            root = tmpdir / "latexpdf"
            assert luigi.build(
                [
                    TestWorkflow(
                        dataset_df=dataset_df_path,
                        result_path=str(root),
                        report_type="latexpdf",
                        specifications_only=True,
                    )
                ],
                local_scheduler=True,
            )

            assert (root / "TestWorkflow_specifications.pdf").exists()
            assert pdfdiff(
                root / "TestWorkflow_specifications.pdf",
                data_dir / "test_report_before_run" / "report_latexpdf.pdf",
                threshold=15,
            )

        @pytest.fixture
        def report_config(self):
            return {
                "project": "Test title",
                "version": "999",
                "author": "Test author",
                "today": "FIXED DATE FOR TESTS",
            }

        def test_rst2pdf_with_config(
            self, tmpdir, dataset_df_path, data_dir, TestWorkflow, report_config
        ):
            root = tmpdir / "rst2pdf"
            assert luigi.build(
                [
                    TestWorkflow(
                        dataset_df=dataset_df_path,
                        result_path=str(root),
                        specifications_only=True,
                        report_config=report_config,
                    )
                ],
                local_scheduler=True,
            )

            assert (root / "TestWorkflow_specifications.pdf").exists()
            assert pdfdiff(
                root / "TestWorkflow_specifications.pdf",
                data_dir / "test_report_before_run" / "report_rst2pdf_with_config.pdf",
            )

        @pytest.mark.skipif(SKIP_IF_NO_LATEXMK, reason=REASON_NO_LATEXMK)
        def test_latexpdf_with_config(
            self, tmpdir, dataset_df_path, data_dir, TestWorkflow, report_config
        ):
            root = tmpdir / "latexpdf"
            assert luigi.build(
                [
                    TestWorkflow(
                        dataset_df=dataset_df_path,
                        result_path=str(root),
                        report_type="latexpdf",
                        specifications_only=True,
                        report_config=report_config,
                    )
                ],
                local_scheduler=True,
            )

            assert (root / "TestWorkflow_specifications.pdf").exists()
            assert pdfdiff(
                root / "TestWorkflow_specifications.pdf",
                data_dir / "test_report_before_run" / "report_latexpdf_with_config.pdf",
            )

        def test_nested_workflows(
            self, tmpdir, dataset_df_path, data_dir, TestNestedWorkflow, report_config
        ):
            root = tmpdir / "rst2pdf_nested"
            assert luigi.build(
                [
                    TestNestedWorkflow(
                        dataset_df=dataset_df_path,
                        result_path=str(root),
                        specifications_only=True,
                        report_config=report_config,
                    )
                ],
                local_scheduler=True,
            )

            assert (root / "TestNestedWorkflow_specifications.pdf").exists()
            assert pdfdiff(
                root / "TestNestedWorkflow_specifications.pdf",
                data_dir / "test_report_before_run" / "report_rst2pdf_nested.pdf",
                threshold=25,
            )
