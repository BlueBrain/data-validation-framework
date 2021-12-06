"""Test the data_validation_framework.rst_tools module."""
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=use-implicit-booleaness-not-comparison
import pytest

from data_validation_framework import rst_tools


def test_init():
    rst = rst_tools.RstFile("test_file")
    assert rst.file_path == "test_file"
    assert rst.mode == "w"
    assert rst._data == []  # pylint: disable=protected-access

    with pytest.raises(ValueError, match="The 'file_path' argument must be a not empty string."):
        rst_tools.RstFile("")

    with pytest.raises(ValueError, match="The 'file_path' argument must be a not empty string."):
        rst_tools.RstFile(None)


@pytest.fixture
def empty_RstFile(tmpdir):
    filename = tmpdir / "test_file.rst"
    return rst_tools.RstFile(str(filename))


def test_add(empty_RstFile):
    empty_RstFile.add("test")
    empty_RstFile.add("test", indent=2)
    empty_RstFile.add("long\nline\nwith\nnewlines", indent=2)
    assert empty_RstFile.get_buffer() == ["test", "  test", "  long\nline\nwith\nnewlines"]


def test_new_line(empty_RstFile):
    empty_RstFile.newline()
    empty_RstFile.newline(3)
    assert empty_RstFile.get_buffer() == ["\n"] * 4

    with pytest.raises(ValueError, match="Count of newlines must be a positive int."):
        empty_RstFile.newline(-1)

    with pytest.raises(ValueError, match="Count of newlines must be a positive int."):
        empty_RstFile.newline(None)


def test_clear(empty_RstFile):
    empty_RstFile.add("test")
    assert empty_RstFile.get_buffer() == ["test"]
    empty_RstFile.clear()
    assert empty_RstFile.get_buffer() == []


def test_str_formatting():
    assert rst_tools.RstFile.bold("test") == "**test**"
    assert rst_tools.RstFile.emph("test") == "*test*"
    assert rst_tools.RstFile.pre("test") == "``test``"
    assert (
        rst_tools.RstFile.inline_link("the text", "http://the.link")
        == "`the text <http://the.link>`_"
    )
    # pylint: disable=protected-access
    assert rst_tools.RstFile._indent("test") == "test"
    assert rst_tools.RstFile._indent("test", 2) == "  test"
    assert rst_tools.RstFile._indent(["test1", "test2"], 2) == ["  test1", "  test2"]


def test_headings(empty_RstFile):
    empty_RstFile.title("The title")
    empty_RstFile.h1("The heading 1")
    empty_RstFile.h2("The heading 2")
    empty_RstFile.h3("The heading 3")
    empty_RstFile.h4("The heading 4")
    empty_RstFile.h5("The heading 5")
    empty_RstFile.h6("The heading 6")
    empty_RstFile.heading_level(3, "The heading 6")

    assert empty_RstFile.get_buffer() == [
        "=========",
        "The title",
        "=========",
        "\n",
        "\n",
        "The heading 1",
        "=============",
        "\n",
        "\n",
        "The heading 2",
        "-------------",
        "\n",
        "\n",
        "The heading 3",
        "~~~~~~~~~~~~~",
        "\n",
        "\n",
        "The heading 4",
        "+++++++++++++",
        "\n",
        "\n",
        "The heading 5",
        "^^^^^^^^^^^^^",
        "\n",
        "\n",
        "The heading 6",
        ";;;;;;;;;;;;;",
        "\n",
        "\n",
        "The heading 6",
        "~~~~~~~~~~~~~",
        "\n",
    ]


def test_toc_tree(empty_RstFile):
    # Default
    empty_RstFile.toc_tree()

    # With max depth
    empty_RstFile.clear()
    empty_RstFile.toc_tree(maxdepth=3)
    assert empty_RstFile.get_buffer() == [
        "\n",
        ".. toctree::",
        "   :maxdepth: 3",
        "   :numbered:",
        "\n",
    ]

    # With not numbered
    empty_RstFile.clear()
    empty_RstFile.toc_tree(numbered=False)
    assert empty_RstFile.get_buffer() == ["\n", ".. toctree::", "\n"]

    # With indent
    empty_RstFile.clear()
    empty_RstFile.toc_tree(indent=2)
    assert empty_RstFile.get_buffer() == ["\n", "  .. toctree::", "     :numbered:", "\n"]

    # With contents
    empty_RstFile.clear()
    empty_RstFile.toc_tree(toc_type="contents")
    assert empty_RstFile.get_buffer() == ["\n", ".. contents::", "\n"]

    # With bad toc type
    empty_RstFile.clear()
    with pytest.raises(ValueError, match="Unknown TOC tree type"):
        empty_RstFile.toc_tree(toc_type="bad type")


def test_code_block(empty_RstFile):
    # Default
    empty_RstFile.code_block()
    assert empty_RstFile.get_buffer() == ["\n", ".. code-block:: ", "\n"]

    # With language
    empty_RstFile.clear()
    empty_RstFile.code_block(language="Python")
    assert empty_RstFile.get_buffer() == ["\n", ".. code-block:: Python", "\n"]


def test_exception(empty_RstFile):
    exception_str = (
        "\t\u001b[0;35mException\u001b[0m "
        "\u001b[0;36m172.18.0.2\u001b[0m "
        "\u001b[0;36m/file/path.test\u001b[0m"
    )

    # Default
    empty_RstFile.exception()
    assert empty_RstFile.get_buffer() == ["\n", "", "\n"]

    # With ANSI escape sequences
    empty_RstFile.clear()
    empty_RstFile.exception(exception_str)
    assert empty_RstFile.get_buffer() == ["\n", "\tException 172.18.0.2 /file/path.test", "\n"]

    # With width
    empty_RstFile.clear()
    empty_RstFile.exception(exception_str, width=15)
    assert empty_RstFile.get_buffer() == ["\n", "Exception", "172.18.0.2", "/file/path.test", "\n"]


def test_list(empty_RstFile):
    list_str = [
        "first",
        "second",
        "third",
    ]

    # Default
    empty_RstFile.list(list_str)
    assert empty_RstFile.get_buffer() == ["* first", "* second", "* third"]

    # With custom bullet
    empty_RstFile.clear()
    empty_RstFile.list(list_str, bullet="-")
    assert empty_RstFile.get_buffer() == ["- first", "- second", "- third"]

    # With empty list
    empty_RstFile.clear()
    empty_RstFile.list([])
    assert empty_RstFile.get_buffer() == []

    # With width
    empty_RstFile.clear()
    empty_RstFile.list([" ".join([i] * 4) for i in list_str], width=15)
    assert empty_RstFile.get_buffer() == [
        "* first first\n  first first",
        "* second second\n  second second",
        "* third third\n  third third",
    ]

    # With width and indent
    empty_RstFile.clear()
    empty_RstFile.list([" ".join([i] * 4) for i in list_str], width=10, indent=2)
    assert empty_RstFile.get_buffer() == [
        "  * first\n    first\n    first\n    first",
        "  * second\n    second\n    second\n    second",
        "  * third\n    third\n    third\n    third",
    ]


def test_write(empty_RstFile):
    empty_RstFile.title("The title")
    empty_RstFile.add("Some text")
    empty_RstFile.write()

    with open(empty_RstFile.file_path, encoding="utf-8") as res_file:
        res = res_file.read()

    assert res == ("=========\n" "The title\n" "=========\n" "\n" "\n" "Some text\n")
