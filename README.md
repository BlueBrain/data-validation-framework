[![Version](https://img.shields.io/pypi/v/data-validation-framework)](https://github.com/BlueBrain/data-validation-framework/releases)
[![Build status](https://github.com/BlueBrain/data-validation-framework/actions/workflows/run-tox.yml/badge.svg?branch=main)](https://github.com/BlueBrain/data-validation-framework/actions)
[![Coverage](https://codecov.io/github/BlueBrain/data-validation-framework/coverage.svg?branch=main)](https://codecov.io/github/BlueBrain/data-validation-framework?branch=main)
[![License](https://img.shields.io/badge/License-Apache%202-blue)](https://github.com/BlueBrain/data-validation-framework/blob/main/LICENSE.txt)
[![Documentation status](https://readthedocs.org/projects/data-validation-framework/badge/?version=latest)](https://data-validation-framework.readthedocs.io/)


# Data Validation Framework

This project provides simple tools to create data validation workflows.
The workflows are based on the [luigi](https://luigi.readthedocs.io/en/stable) library.

The main objective of this framework is to gather in a same place both the specifications that the
data must follow and the code that actually tests the data. This avoids having multiple documents
to store the specifications and a repository to store the code.


## Installation

This package should be installed using pip:

```bash
pip install data-validation-framework
```

## Usage

### Building a workflow

Building a new workflow is simple, as you can see in the following example:

```python
import luigi
import data_validation_framework as dvf


class ValidationTask1(dvf.task.ElementValidationTask):
    """Use the class docstring to describe the specifications of the ValidationTask1."""

    output_columns = {"col_name": None}

    @staticmethod
    def validation_function(row, output_path, *args, **kwargs):
        # Return the validation result for one row of the dataset
        if row["col_name"] <= 10:
            return dvf.result.ValidationResult(is_valid=True)
        else:
            return dvf.result.ValidationResult(
                is_valid=False,
                ret_code=1,
                comment="The value should always be <= 10"
            )


def external_validation_function(df, output_path, *args, **kwargs):
    # Update the dataset inplace here by setting values to the 'is_valid' column.
    # The 'ret_code' and 'comment' values are optional, they will be added to the report
    # in order to help the user to understand why the dataset did not pass the validation.

    # We can use the value from kwargs["param_value"] here.
    if int(kwargs["param_value"]) <= 10:
        df["is_valid"] = True
    else:
        df["is_valid"] = False
        df["ret_code"] = 1
        df["comment"] = "The value should always be <= 10"


class ValidationTask2(dvf.task.SetValidationTask):
    """In some cases you might want to keep the docstring to describe what a developper
    needs to know, not the end-user. In this case, you can use the ``__specifications__``
    attribute to store the specifications."""

    a_parameter = luigi.Parameter()

    __specifications__ = """Use the __specifications__ to describe the specifications of the
    ValidationTask2."""

    def inputs(self):
        return {ValidationTask1(): {"col_name": "new_col_name_in_current_task"}}

    def kwargs(self):
        return {"param_value": self.a_parameter}

    validation_function = staticmethod(external_validation_function)


class ValidationWorkflow(dvf.task.ValidationWorkflow):
    """Use the global workflow specifications to give general context to the end-user."""

    def inputs(self):
        return {
            ValidationTask1(): {},
            ValidationTask2(): {},
        }
```

Where the `ValidationWorkflow` class only defines the sub-tasks that should be called for the
validation. The sub-tasks can be either a `dvf.task.ElementValidationTask` or a
`dvf.task.SetValidationTask`. In both cases, you can define relations between these sub-tasks
since one could need the result of another one to run properly. This is defined in two steps:

1. in the required task, a `output_columns` attribute should be defined so that the next tasks
   can know what data is available, as shown in the previous example for the `ValidationTask1`.
2. in the task that requires another task, a `inputs` method should be defined, as shown in the
   previous example for the `ValidationTask2`.

The sub-classes of `dvf.task.ElementValidationTask` should return a
`dvf.result.ValidationResult` object. The sub-classes of `dvf.task.SetValidationTask` should
return a `Pandas.DataFrame` object with at least the following columns
`["is_valid", "ret_code", "comment", "exception"]` and with the same index as the input dataset.

## Generate the specifications of a workflow

The specifications that the data should follow can be generated with the following luigi command:

```bash
luigi --module test_validation ValidationWorkflow --log-level INFO --local-scheduler --result-path out --ValidationTask2-a-parameter 15 --specifications-only
```

## Running a workflow

The workflow can be run with the following luigi command (note that the module `test_validation`
must be available in your `sys.path`):


```bash
luigi --module test_validation ValidationWorkflow --log-level INFO --local-scheduler --dataset-df dataset.csv --result-path out --ValidationTask2-a-parameter 15
```

This workflow will generate the following files:

* `out/report_ValidationWorkflow.pdf`: the PDF validation report.
* `out/ValidationTask1/report.csv`: The CSV containing the validity values of the task
  `ValidationTask1`.
* `out/ValidationTask2/report.csv`: The CSV containing the validity values of the task
  `ValidationTask2`.
* `out/ValidationWorkflow/report.csv`: The CSV containing the validity values of the complete
  workflow.

.. note::

    As any `luigi <https://luigi.readthedocs.io/en/stable>`_ workflow, the values can be stored
    into a `luigi.cfg` file instead of being passed to the CLI.

## Advanced features

### Require a regular Luigi task

In some cases, one want to execute a regular Luigi task in a validation workflow. In this case, it
is possible to use the `extra_requires()` method to pass these extra requirements. In the
validation task it is then possible to get the targets of these extra requirements using the
`extra_input()` method.

```python
class TestTaskA(luigi.Task):

    def run(self):
        # Do something and write the 'target.file'

    def output(self):
        return target.OutputLocalTarget("target.file")

class TestTaskB(task.SetValidationTask):

    output_columns = {"extra_target_path": None}

    def kwargs(self):
        return {"extra_task_target_path": self.extra_input().path}

    def extra_requires(self):
        return TestTaskA()

    @staticmethod
    def validation_function(df, output_path, *args, **kwargs):
        df["is_valid"] = True
        df["extra_target_path"] = kwargs["extra_task_target_path"]
```

## Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project, a research
center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH
Board of the Swiss Federal Institutes of Technology.

For license and authors, see `LICENSE.txt` and `AUTHORS.md` respectively.

Copyright © 2021 Blue Brain Project/EPFL
