"""Specific tasks."""
import logging
import sys
import time
import traceback
import warnings
from functools import partial
from pathlib import Path

import luigi
import pandas as pd
from luigi.parameter import ChoiceParameter
from luigi.parameter import DictParameter
from luigi.task import flatten as task_flatten
from luigi_tools.parameter import BoolParameter
from luigi_tools.parameter import OptionalBoolParameter
from luigi_tools.parameter import OptionalIntParameter
from luigi_tools.parameter import OptionalPathParameter
from luigi_tools.parameter import OptionalStrParameter
from luigi_tools.task import LogTargetMixin
from luigi_tools.task import RerunMixin
from numpy import VisibleDeprecationWarning

from data_validation_framework.report import make_report
from data_validation_framework.result import ValidationResult
from data_validation_framework.result import ValidationResultSet
from data_validation_framework.target import ReportTarget
from data_validation_framework.target import TaggedOutputLocalTarget
from data_validation_framework.util import apply_to_df

L = logging.getLogger(__name__)
INDEX_LABEL = "__index_label__"
SKIP_COMMENT = "Skipped by user."


class ValidationError(Exception):
    """Exception raised if the validation could not be performed properly."""


class TagResultOutputMixin:
    """Initialize target prefixes and optionally add a tag to the resut directory.

    .. warning::

        If this mixin is used alongside the :class:`luigi_tools.task.RerunMixin`, then it can have
        two different behaviors:

        * if placed on the right side of the `RerunMixin`, a new tag is create and thus rerun does
          not do anything.
        * if placed on the left side of the `RerunMixin`, the untagged directory is remove then a
          tagged directory is created.
    """

    tag_output = BoolParameter(
        default=False, description=":bool: Add a tag suffix to the output directory."
    )
    result_path = OptionalPathParameter(
        default=None, description=":str: Path to the global output directory."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def num_tag(path, num):
            return path.with_name(f"{path.name}_{num}")

        if self.tag_output:
            path = Path(f"{self.result_path}_{time.strftime('%Y%m%d-%Hh%Mm%Ss')}")
            tagged_output = path
            num = 0
            while tagged_output.exists():
                L.debug("Tagged output '%s' already exists", tagged_output)
                num += 1
                tagged_output = num_tag(path, num)

            if num > 0 and RerunMixin in self.__class__.mro() and self.rerun:
                warnings.warn(
                    "Using 'rerun' with conflicting tag output results in creating a new tag or "
                    "removing the untagged result directory (depending on the inheritance order).",
                )

            L.info("Tagged output is: %s", tagged_output)
            self.result_path = tagged_output

        if self.result_path is not None:
            TaggedOutputLocalTarget.set_default_prefix(self.result_path)


class BaseValidationTask(LogTargetMixin, RerunMixin, TagResultOutputMixin, luigi.Task):
    """Base luigi task used for validation steps.

    .. warning::
        Only use child of this class such as :class:`ElementValidationTask` or
        :class:`SetValidationTask`.

    To provide a validation function, use it as:

    .. code-block:: python

        class Task(ChildBaseValidationTask):
            data_dir = luigi.Parameter(default='my_data')
            validation_function = staticmethod(my_validation_function)
            output_columns = {'my_new_col': None}

            def inputs(self):
                return {'PreviousTask(): {"input_data", "input_data"}

    .. note::

        The ``dataset_df`` and ``result_path`` attributes are passed to the requirements when they
        have ``None`` values for these attributes.

    """

    # I/O Parameters
    dataset_df = OptionalPathParameter(
        description=":str: Path to the input dataset.",
        default=None,
        exists=True,
    )
    input_index_col = OptionalStrParameter(
        default=None, description=":str: Name of the column used as index."
    )
    data_dir = OptionalStrParameter(
        default="data",
        description=(
            ":str: name of folder to store addittional files created by a task (the provided "
            "validation function must take this as argument)."
        ),
    )

    # Naming Parameters
    custom_task_name = OptionalStrParameter(
        default=None,
        significant=False,
        description=(
            ":str: An optional custom name given to the task (the class name is used if not "
            "provided)."
        ),
    )

    # User attributs
    output_columns = None
    """dict: A dict with names as keys and empty values as values for new columns created by the
    current task.
    """

    gather_inputs = False
    """bool: Gather the inputs in the :class:`pandas.DataFrame` given to the
    :meth:`validation_function`.
    """

    # Internal attributs
    nb_total = None
    """int: Total number of processed elements."""
    nb_valid = None
    """int: Total number of valid elements."""
    results = None
    """pandas.DataFrame: The results of the :meth:`validation_function`."""

    nb_processes = OptionalIntParameter(
        default=None,
        description=":int: The number of parallel processes to use.",
        significant=False,
    )

    redirect_stdout = OptionalBoolParameter(
        default=None,
        parsing=luigi.BoolParameter.EXPLICIT_PARSING,
        description=(
            ":bool: Capture stdout from the validation function to make it work with progress "
            "bar. Disable it if you want to use PDB inside the validation function."
        ),
        significant=False,
    )

    def __init__(self, *args, **kwargs):
        warnings.filterwarnings("ignore", module="numpy", category=VisibleDeprecationWarning)
        warnings.filterwarnings("ignore", module="luigi.task", category=DeprecationWarning)
        event_handler = super().event_handler

        # pylint: disable=unused-variable
        @event_handler(luigi.Event.SUCCESS)
        def success_summary(self):
            """Summary report of the task."""
            L.info("==========================================")
            task_summary = f"SUMMARY {self.task_name}: {self.nb_valid} / {self.nb_total} passed"
            L.info(task_summary)
            L.info("==========================================")

        super().__init__(*args, **kwargs)

        self._report_cols = list(ValidationResultSet.out_cols.keys())

    def inputs(self):
        """Information about required input data.

        This method can be overridden and should return a dict of the following form:

        .. code-block:: python

            {<task_name>(): {"<input_column_name>": "<current_column_name>"}}

        where:
            - ``<task_name>`` is the name of the required task,
            - ``<input_column_name>`` is the name of the column we need from the report of
              task_name,
            - ``<current_column_name>`` is the name of the same column in the current task.
        """
        # pylint: disable=no-self-use
        return None

    def args(self):
        """List of addititonal arguments to provide to :meth:`validation_function`.

        This method can be overridden to pass these arguments to the :meth:`validation_function`
        and to the :meth:`pre_process` and :meth:`post_process` methods.
        """
        # pylint: disable=no-self-use
        return []

    def kwargs(self):
        """Dict of addititonal keyword arguments to provide to :meth:`validation_function`.

        This method can be overridden to pass these keyword arguments to the
        :meth:`validation_function` and to the :meth:`pre_process` and
        :meth:`post_process` methods.
        """
        # pylint: disable=no-self-use
        return {}

    def read_dataset(self):
        """Import the dataset to a :class:`pandas.DataFrame`.

        This method can be overriden to load custom data (e.g. GeoDataFrame, etc.).
        The dataset should alway be loaded from the path given by `self.dataset_df`.
        """
        return pd.read_csv(self.dataset_df, index_col=self.input_index_col)

    def pre_process(self, df, args, kwargs):
        """Method executed before applying the external function."""

    def post_process(self, df, args, kwargs):
        """Method executed after applying the external function."""

    def requires(self):
        """Process the inputs to generate the requirements."""
        if self.inputs():
            requires = list(self.inputs().keys())  # pylint: disable=not-callable
            for req in requires:
                if req.dataset_df is None:
                    req.dataset_df = self.dataset_df
                    req.input_index_col = self.input_index_col
                if req.result_path is None:
                    req.result_path = self.result_path
                if req.nb_processes is None:
                    req.nb_processes = self.nb_processes
                if req.redirect_stdout is None:
                    req.redirect_stdout = self.redirect_stdout
        else:
            requires = []
        return requires + task_flatten(self.extra_requires())

    def extra_requires(self):
        """Requirements that should not be considered as validation tasks."""
        # pylint: disable=no-self-use
        return []

    def extra_input(self):
        """Targets of the tasks given to extra_requires()."""
        return luigi.task.getpaths(self.extra_requires())

    @staticmethod
    def _rename_cols(df):
        if df.columns.nlevels > 1:
            df.columns = pd.MultiIndex.from_tuples(
                [tuple("" if "Unnamed: " in j else j for j in i) for i in df.columns.tolist()]
            )
        return df

    def _process(self, df, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Please do not use the BaseValidationTask but deriving class.")

    @staticmethod
    def validation_function(*args, **kwargs):  # pragma: no cover
        """The validation function to apply to the current data set."""
        raise ValueError(
            "You must set the 'validation_function' attribute of the class to a reference to a "
            "function."
        )

    def __getattribute__(self, name):
        """Unbound the validation function if it is bounded to the current instance."""
        if name == "validation_function":
            func = super().__getattribute__(name)
            if hasattr(func, "__func__"):
                return func.__func__
            return func

        return super().__getattribute__(name)

    @staticmethod
    def check_inputs(inputs):
        """Check a given dict of inputs."""
        if not inputs:
            return False

        # Check column names conflicts
        forbidden_cols = ValidationResultSet.out_cols.keys()
        col_names = {}
        for t, col_mapping in inputs.items():
            for new_col in col_mapping.values():
                checked = col_names.get(new_col)
                if checked is not None:
                    raise ValueError(
                        f"The destination column '{new_col}' of the tasks {t} and {checked} are "
                        "conflicting."
                    )
                if new_col in forbidden_cols:
                    raise ValueError(
                        f"The destination column of the task {t} can not be one of "
                        f"{list(forbidden_cols)}."
                    )
                col_names[new_col] = t

        # Check if column mapping was found
        if not col_names:
            return False
        return True

    @staticmethod
    def filter_columns(all_dfs, mappings):
        """Filter the columns of the given DataFrames according to a given mapping."""
        filtered_dfs = []
        for t, df in all_dfs.items():
            col_mapping = mappings.get(t, {})

            # Pick and rename columns
            try:
                filtered_dfs.append(df[col_mapping.keys()].rename(columns=col_mapping))
            except KeyError as e:
                missing_cols = sorted(set(col_mapping.keys()) - set(df.columns))
                raise KeyError(
                    f"The columns {missing_cols} are missing from the output columns of {t}."
                ) from e

        return filtered_dfs

    def _get_dataset(self):
        """Get the dataset."""
        if self.dataset_df is not None:
            L.info("Input dataset: %s", Path(self.dataset_df).resolve())
            new_df = self.read_dataset()
            duplicated_index = new_df.index.duplicated()
            if duplicated_index.any():
                raise IndexError(
                    "The following index values are duplicated: "
                    f"{sorted(new_df.index.to_series().loc[duplicated_index])}"
                )
        else:
            new_df = pd.DataFrame()

        return new_df

    def _join_inputs(self, new_df):
        """Get the inputs and join them to the dataset."""
        if self.inputs():
            # Check inputs
            with_mapping = self.check_inputs(self.inputs())  # pylint: disable=not-callable

            # Get the input targets and their DataFrames
            all_inputs = {
                i: [
                    target
                    for target in task_flatten(i.output())
                    if isinstance(target, ReportTarget)
                ]
                for i in task_flatten(self.inputs().keys())
            }
            all_report_paths = {
                t: [r.path for r in reports][0] for t, reports in all_inputs.items()
            }
            L.debug("Importing the following reports: %s", all_report_paths)
            all_dfs = {
                task_obj: self._rename_cols(
                    pd.read_csv(path, index_col=INDEX_LABEL).rename_axis(index="index")
                )
                for task_obj, path in all_report_paths.items()
            }

            # Compute the global is_valid column
            are_valid = pd.concat([i["is_valid"] for i in all_dfs.values()], axis=1)
            is_valid = are_valid.all(axis=1)
            new_df["is_valid"] = is_valid

            # Filter columns in the DataFrames
            if with_mapping:
                # pylint: disable=not-callable
                filtered_dfs = self.filter_columns(all_dfs, self.inputs())

                # Concatenate all DataFrames
                filtered_df = pd.concat(filtered_dfs, axis=1)

                # Erase columns
                new_df[filtered_df.columns] = filtered_df

                # Add filtered columns to the report ones
                self._report_cols.extend(
                    [col for col in filtered_df.columns if col not in self._report_cols]
                )

            if self.gather_inputs:
                # Use column MultiIndex on each input DataFrame in order to gather them
                for task_obj, df in all_dfs.items():
                    df.columns = pd.MultiIndex.from_tuples(
                        [(task_obj.task_name, i) for i in df.columns.values]
                    )

                # Concatenate all DataFrames
                gathered = pd.concat(all_dfs.values(), axis=1)
                gathered.columns = gathered.columns.to_flat_index()

                # Add the gathered DataFrame to the new DataFrame
                new_df = new_df.join(gathered)

                # Add gathered columns to the report ones
                self._report_cols.extend(
                    [col for col in gathered.columns if col not in self._report_cols]
                )

            # Clean inconsistent indexes
            null_index = new_df["is_valid"].isnull()
            if null_index.any():
                L.warning(
                    "The following inconsistent indexes between the dataset and the inputs are "
                    "ignored: %s",
                    null_index.loc[null_index].index.tolist(),
                )
            new_df.dropna(subset=["is_valid"], inplace=True)

        return new_df

    def run(self):
        """The main process of the current task."""
        # Import the DataFrame(s)
        if self.dataset_df is None and self.inputs() is None:
            raise ValueError("Either the 'dataset_df' parameter or a requirement must be provided.")

        new_df = self._get_dataset()
        new_df = self._join_inputs(new_df)

        # Create the output directory
        self.output()["data"].pathlib_path.mkdir(parents=True, exist_ok=True)

        # Get args
        if callable(self.args):
            args = self.args()
        else:
            args = self.args
        if not isinstance(args, (list, tuple)):
            raise TypeError(
                "The 'args' must either be a method returning a list or a tuple, or an actual "
                "list or tuple."
            )

        # Get kwargs
        if callable(self.kwargs):
            kwargs = self.kwargs()
        else:
            kwargs = self.kwargs
        if not isinstance(kwargs, dict):
            raise TypeError(
                "The 'kwargs' must either be a method returning a dict, or an actual dict."
            )

        # Add output columns
        if self.output_columns:
            self._report_cols.extend(
                [col for col in self.output_columns if col not in self._report_cols]
            )

        # Format the DataFrame
        df = ValidationResultSet(new_df, output_columns=self.output_columns)

        # Copy current index
        index = df.index.copy()

        # Apply pre process
        self.pre_process(df, args, kwargs)

        # Apply the validation function to the DataFrame
        res = self._process(df, *args, **kwargs)

        # Apply post process
        self.post_process(res, args, kwargs)

        # Count valid items
        self.nb_valid = res["is_valid"].sum()
        self.nb_total = len(res.index)
        self.results = res[list(ValidationResultSet.out_cols.keys())]

        # Export the DataFrame to CSV file
        res[self._report_cols].to_csv(
            self.output()["report"].path, index=True, index_label=INDEX_LABEL
        )

        # Check that the index was not modified
        if not res.index.equals(index):
            raise IndexError(
                "The index changed during the process. Please update your validation function or "
                "your pre/post process functions to avoid this behaviour."
            )

    def output(self):
        """The targets of the current task."""
        class_path = Path(self.task_name)
        prefix = None if self.result_path is None else self.result_path.absolute()
        return {
            "report": ReportTarget(
                class_path / "report.csv",
                prefix=prefix,
                create_parent=False,  # Do not create the parent here because of the tagged output
                task_name=self.task_name,
            ),
            "data": TaggedOutputLocalTarget(
                class_path / self.data_dir,
                prefix=prefix,
                create_parent=False,  # Do not create the parent here because of the tagged output
            ),
        }

    @property
    def task_name(self):
        """The name of the task."""
        return self.custom_task_name or self.task_family

    @property
    def __specifications__(self):
        """A specific docstring only used in the report.

        If not overridden, the report uses the usual docstring.
        """
        return self.__doc__


class ElementValidationTask(BaseValidationTask):
    """A class to validate each element of a data set without considering the other elements.

    The ``validation_function`` will receive the row and the output_path as first arguments.
    """

    def _process(self, df, *args, **kwargs):
        return apply_to_df(
            df,
            self.validation_function,
            self.output()["data"].pathlib_path,
            *args,
            nb_processes=self.nb_processes,
            redirect_stdout=self.redirect_stdout,
            **kwargs,
        )


class SetValidationTask(BaseValidationTask):
    """A class to validate an entire data set (usefull for global properties).

    The ``validation_function`` will receive the DataFrame and the output_path as first arguments.

    .. note::

        The given :class:`pandas.DataFrame` will always have the columns
        ``["is_valid", "ret_code", "comment", "exception"]``. The ``validation_function``
        should at least update the values for the ``is_valid`` column.
    """

    def _process(self, df, *args, **kwargs):
        try:
            # pylint: disable=not-callable
            self.validation_function(df, self.output()["data"].pathlib_path, *args, **kwargs)
        except Exception:  # pylint: disable=broad-except
            exception = "".join(traceback.format_exception(*sys.exc_info()))
            L.warning("Exception for entire set: %s", exception)
            df["exception"] = exception
            df["is_valid"] = False
            df["ret_code"] = 1

        # Check that return codes are consistent with validity values
        if (
            not df.loc[(df["is_valid"]) & (df["ret_code"] == 1)].empty
            or not df.loc[(~df["is_valid"]) & (df["ret_code"] == 0)].empty
        ):
            raise ValueError("The 'ret_code' values are not consistent with the 'is_valid' values.")
        if not df.loc[(df["comment"].isnull()) & (~df["ret_code"].isin([0, 1]))].empty:
            warnings.warn("A comment should be set when the 'ret_code' is greater than 1.")

        # Fix missing values for return codes according to validity values
        df.loc[(df["is_valid"]) & (df["ret_code"].isnull()), "ret_code"] = 0
        df.loc[(~df["is_valid"]) & (df["ret_code"].isnull()), "ret_code"] = 1

        # Fix missing comment and exception values
        df.loc[df["comment"].isnull(), "comment"] = ""
        df.loc[df["exception"].isnull(), "exception"] = ""

        return df


class ValidationWorkflow(SetValidationTask):
    """Class to define and process a validation workflow."""

    report_path = OptionalStrParameter(
        default=None, description=":str: Path to the workflow report."
    )
    generate_report = BoolParameter(
        default=True, description=":bool: Trigger the report generation."
    )
    report_type = ChoiceParameter(
        default="pdf",
        description=":str: Type of report ('pdf': basic, 'latexpdf': beautiful).",
        choices=["pdf", "latexpdf"],
    )
    report_config = DictParameter(
        default=None,
        description=":dict: The configuration used by Sphinx to build the report.",
    )
    specifications_only = BoolParameter(
        default=False,
        description=":bool: Only outputs the dataset specification document.",
    )

    gather_inputs = True

    def __init__(self, *args, **kwargs):
        event_handler = super().event_handler

        # pylint: disable=unused-variable
        @event_handler(luigi.Event.SUCCESS)
        def spec_report(current_task):
            """Hook to create a specification report."""
            L.debug("Generating report of %s", current_task)
            if current_task.generate_report:
                try:
                    make_report(current_task, config=self.report_config)
                # pylint: disable=broad-except
                except Exception as e:  # pragma: no cover
                    L.error(
                        "The report could not be generated because of the following exception: %s",
                        e,
                    )

        super().__init__(*args, **kwargs)

        if self.specifications_only:
            if not self.report_path:
                self.report_path = f"{self.task_name}_specifications.pdf"
            make_report(self, use_data=False, config=self.report_config)

            self.complete = lambda: True

    @staticmethod
    def validation_function(*args, **kwargs):
        """The validation function to apply to the current data set .

        This method should usually do nothing for :class:`ValidationWorkflow` as this class is only
        supposed to gather validation steps.
        """


def _skippable_element_validation_function(validation_function, skip, *args, **kwargs):
    """Skipping wrapper for an element validation function."""
    if skip:
        return ValidationResult(is_valid=True, comment=SKIP_COMMENT)
    return validation_function(*args, **kwargs)


def _skippable_set_validation_function(validation_function, skip, *args, **kwargs):
    """Skipping wrapper for a set validation function."""
    df = kwargs.get("df", args[0])
    if skip:
        df.loc[df["is_valid"], "comment"] = SKIP_COMMENT
    else:
        validation_function(*args, **kwargs)


def SkippableMixin(default_value=False):
    """Create a mixin class to add a ``skip`` parameter.

    This mixin must be applied to a :class:`data_validation_framework.ElementValidationTask`.
    It will create a ``skip`` parameter and wrap the validation function to just skip it if the
    ``skip`` argument is set to ``True``. If skipped, it will keep the ``is_valid`` values as is and
    add a specific comment to inform the user.

    Args:
        default_value (bool): The default value for the ``skip`` argument.
    """

    class Mixin:
        """A mixin to add a ``skip`` parameter to a :class:`luigi.task`."""

        skip = BoolParameter(default=default_value, description=":bool: Skip the task")

        def __init__(self, *args, **kwargs):

            super().__init__(*args, **kwargs)

            if isinstance(self, ElementValidationTask):
                new_validation_function = partial(
                    _skippable_element_validation_function,
                    self.validation_function,
                    self.skip,
                )
            elif isinstance(self, SetValidationTask) and not isinstance(self, ValidationWorkflow):
                new_validation_function = partial(
                    _skippable_set_validation_function,
                    self.validation_function,
                    self.skip,
                )
            else:
                raise TypeError(
                    "The SkippableMixin can only be associated with childs of ElementValidationTask"
                    " or SetValidationTask"
                )
            self._skippable_validation_function = self.validation_function
            self.validation_function = new_validation_function

    return Mixin
