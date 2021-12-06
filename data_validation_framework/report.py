"""Specific tasks."""
import copy
import json
import logging
import os
import shutil
import tempfile
import warnings
from collections import defaultdict
from inspect import cleandoc
from pathlib import Path

import sphinx.cmd.build  # pylint: disable=import-error
import sphinx.locale  # pylint: disable=import-error
from luigi_tools.util import get_dependency_graph

from data_validation_framework.rst_tools import RstFile

L = logging.getLogger(__name__)


_DEFAULT_REPORT_CONFIG = {
    "project": "Data validation report",
    "version": "1",
    "extensions": [
        "sphinx.ext.graphviz",
        "sphinx.ext.intersphinx",
        "sphinx.ext.napoleon",
        "sphinx.ext.todo",
        "rst2pdf.pdfbuilder",
    ],
    "author": "",
    "intersphinx_mapping": {
        "python": ("https://docs.python.org/3", None),
        "luigi": ("https://luigi.readthedocs.io/en/stable", None),
    },
}


def build_report_conf(dest, **kwargs):
    """Build the `conf.py` file use by Sphinx."""
    config = copy.deepcopy(_DEFAULT_REPORT_CONFIG)
    project = kwargs.pop("project", _DEFAULT_REPORT_CONFIG["project"])
    author = kwargs.pop("author", _DEFAULT_REPORT_CONFIG["author"])

    extensions = kwargs.pop("extensions", [])
    extensions.extend(config["extensions"])
    extensions = list(set(extensions))
    config["extensions"] = extensions

    config.update(
        {
            "pdf_documents": [
                ("index", "report", project, author),
            ],
            "latex_documents": [
                ("index", "report.tex", project, author, "howto"),
            ],
        }
    )

    config["intersphinx_mapping"].update(kwargs.pop("intersphinx_mapping", {}))

    config.update(kwargs)

    with open(dest, "w", encoding="utf-8") as config_file:
        config_file.writelines(["null = None\n"])
        config_file.writelines([f"{i[0]} = {json.dumps(i[1])} \n" for i in config.items()])


def build_subtree(node, deps, known_nodes=None):
    """Build the deps subtree of a given task."""
    if known_nodes is None:
        known_nodes = set()
    subtree = {}
    known_nodes.add(node)
    for t in deps[node]:
        if t not in known_nodes:
            subtree[t] = {}
            known_nodes.add(t)
    for t in subtree:
        subtree[t] = build_subtree(t, deps, known_nodes)
    return subtree


def description_block(row, rst_file, indent=0, max_length=100):
    """Create a block for a failed feature."""
    list_indent = indent + 4
    rst_file.list([row.name], indent=indent)
    bullets = [
        f"return code: {row.ret_code}",
        f"comment: {row.comment}",
    ]
    if row.exception:
        bullets.append("exception:")
    rst_file.list(
        bullets,
        indent=list_indent,
        width=max_length - list_indent,
    )
    if row.exception:
        rst_file.code_block("Bash", indent + 8)
        exception_indent = indent + 12
        exception_width = max(20, max_length - exception_indent - 21)
        rst_file.exception(row.exception, exception_indent, width=exception_width)


def build_rst(rst_file, tree, level=1, use_data=True):
    """Build the RST file that can then be built with Sphinx."""
    for task_obj, subtree in tree.items():
        sublevel = level + 1
        rst_file.heading_level(level, task_obj.task_name)
        rst_file.add(cleandoc(task_obj.__specifications__))
        if use_data and task_obj.results is not None:
            df = task_obj.results.copy()
            succeeded = df.loc[df["is_valid"]]
            if len(succeeded) > 0:
                without_comment = succeeded.loc[
                    (succeeded["comment"].isnull()) | (succeeded["comment"] == "")
                ]
                if len(without_comment) > 0:
                    rst_file.heading_level(sublevel, "Validated features")
                    rst_file.add(", ".join(sorted(without_comment.index.astype(str).tolist())))

                with_comment = succeeded.loc[
                    (~succeeded["comment"].isnull()) & (succeeded["comment"] != "")
                ]
                if len(with_comment) > 0:
                    rst_file.heading_level(sublevel, "Validated features with warnings")
                    with_comment.apply(lambda x: description_block(x, rst_file), axis=1)
            if subtree:
                failed = df.loc[~df["is_valid"]]
                if len(failed) > 0:
                    rst_file.heading_level(sublevel, "Failed features")
                    rst_file.add(", ".join(sorted(failed.index.astype(str).tolist())))
            else:
                failed = df.loc[(~df["is_valid"]) & (~df["ret_code"].isnull())].sort_index()
                if len(failed) > 0:
                    rst_file.heading_level(sublevel, "Failed features")
                    failed.apply(lambda x: description_block(x, rst_file), axis=1)

        if subtree:
            rst_file.heading_level(sublevel, "Sub tasks")
            build_rst(rst_file, subtree, sublevel + 1)

        rst_file.newline()


# pylint: disable=too-many-statements
def make_report(root, use_data=True, config=None):
    """Create the report based on the current workflow."""
    g = get_dependency_graph(root)

    deps = defaultdict(list)
    for parent, child in g:
        deps[parent].append(child)

    tree = {root: build_subtree(root, deps)}

    # Build the PDF in a temporary directory then move the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = Path(tmpdirname)
        source_path = path / "source_report"
        build_path = path / "build_report"
        source_path.mkdir(parents=True, exist_ok=True)
        build_report_conf(source_path / "conf.py", **(config or {}))

        # Set environment variable to make latexmk quiet (only used when report_type == latexpdf)
        os.environ["LATEXMKOPTS"] = "-quiet -f"

        # Build the RST buffer
        rst_file = RstFile(source_path / "index.rst")
        rst_file.title("Validation report")
        rst_file.toc_tree(maxdepth=5, numbered=True)

        build_rst(rst_file, tree, use_data=use_data)

        # Write the RST file
        rst_file.write()

        # Build the PDF file from the RST file
        # Run the following Sphinx command:
        # sphinx-build -M latexpdf <source_report> <build_report> -W
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sphinx.cmd.build.make_main(
                ["-M", root.report_type, str(source_path), str(build_path), "-q"]
            )

        # Copy the report to its final destination
        report_path = root.report_path or f"report_{root.task_name}.pdf"
        if root.result_path is not None:  # pragma: no cover
            report_path = Path(root.result_path) / report_path

        if root.report_type == "pdf":
            input_path = build_path / "pdf" / "report.pdf"
        elif root.report_type == "latexpdf":  # pragma: no cover
            input_path = build_path / "latex" / "report.pdf"
        else:  # pragma: no cover
            input_path = build_path / root.report_type / "report.pdf"

        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(input_path, report_path)
        L.info("The report was generated here: %s", report_path)
