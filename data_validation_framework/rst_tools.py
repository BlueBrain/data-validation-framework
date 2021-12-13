"""Some tools to generate RST files."""
import re
import textwrap


class RstFile:
    """Class to help creating proper RST files."""

    def __init__(self, file_path, mode="w"):
        if not file_path:
            raise ValueError("The 'file_path' argument must be a not empty string.")
        self.file_path = file_path
        self.mode = mode
        self._data = []
        pattern = r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]"
        self._ANSI_re_compiled = re.compile(pattern)

    def add(self, content, indent=0):
        """Add a single line to the internal buffer."""
        if isinstance(content, list):
            self._data.extend(self._indent(content, indent=indent))
        else:
            self._data.append(self._indent(content, indent=indent))

    def newline(self, count=1):
        """Add an empty line to the internal buffer."""
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Count of newlines must be a positive int.")

        self.add(["\n"] * count)

    def write(self):
        """Write the internal buffer into the given file."""
        with open(self.file_path, self.mode, encoding="utf-8") as f:
            f.write("\n".join(self._data))
            f.write("\n")

    def get_buffer(self):
        """Get the internal buffer."""
        return self._data

    def clear(self):
        """Clear the internal buffer."""
        self._data = []

    @staticmethod
    def bold(string):
        """Bold the given string."""
        return f"**{string}**"

    @staticmethod
    def emph(string):
        """Emph the given string."""
        return f"*{string}*"

    @staticmethod
    def pre(string):
        """Pre the given string."""
        return f"``{string}``"

    @staticmethod
    def inline_link(text, link):
        """Create a link from the given string and URL."""
        return f"`{text} <{link}>`_"

    @staticmethod
    def _indent(content, indent=0):
        """Indent the given string."""
        if indent == 0:
            return content
        indent = " " * indent
        if isinstance(content, list):
            return ["".join([indent, line]) for line in content]
        return "".join([indent, content])

    def title(self, text, indent=0, char="="):
        """Add a title to the internal buffer."""
        line = char * len(text)
        self.add([line, text, line], indent=indent)
        self.newline()

    def toc_tree(self, toc_type="toctree", maxdepth=None, numbered=True, indent=0):
        """Add a toc tree to the internal buffer."""
        self.newline()

        if toc_type == "toctree":
            self.add(".. toctree::", indent=indent)

            if maxdepth:
                self.add(f"   :maxdepth: {maxdepth}", indent=indent)

            if numbered:
                self.add("   :numbered:", indent=indent)

        elif toc_type == "contents":
            self.add(".. contents::", indent=indent)

        else:
            raise ValueError("Unknown TOC tree type")

        self.newline()

    def heading(self, text, char, indent=0):
        """Add a heading to the internal buffer."""
        self.newline()
        self.add([text, char * len(text)], indent=indent)
        self.newline()

    def h1(self, text, indent=0):
        """Add a heading of level 1 to the internal buffer."""
        self.heading(text, char="=", indent=indent)

    def h2(self, text, indent=0):
        """Add a heading of level 2 to the internal buffer."""
        self.heading(text, char="-", indent=indent)

    def h3(self, text, indent=0):
        """Add a heading of level 3 to the internal buffer."""
        self.heading(text, char="~", indent=indent)

    def h4(self, text, indent=0):
        """Add a heading of level 4 to the internal buffer."""
        self.heading(text, char="+", indent=indent)

    def h5(self, text, indent=0):
        """Add a heading of level 5 to the internal buffer."""
        self.heading(text, char="^", indent=indent)

    def h6(self, text, indent=0):
        """Add a heading of level 6 to the internal buffer."""
        self.heading(text, char=";", indent=indent)

    def h7(self, text, indent=0):
        """Add a heading of level 7 to the internal buffer."""
        self.heading(text, char="_", indent=indent)

    def h8(self, text, indent=0):
        """Add a heading of level 8 to the internal buffer."""
        self.heading(text, char="*", indent=indent)

    def h9(self, text, indent=0):
        """Add a heading of level 9 to the internal buffer."""
        self.heading(text, char="#", indent=indent)

    def h10(self, text, indent=0):
        """Add a heading of level 10 to the internal buffer."""
        self.heading(text, char=">", indent=indent)

    def h11(self, text, indent=0):
        """Add a heading of level 11 to the internal buffer."""
        self.heading(text, char="<", indent=indent)

    def h12(self, text, indent=0):
        """Add a heading of level 12 to the internal buffer."""
        self.heading(text, char=",", indent=indent)

    def heading_level(self, level, *args, **kwargs):
        """Add a heading of given level to the internal buffer."""
        level_headings = {
            0: self.title,
            1: self.h1,
            2: self.h2,
            3: self.h3,
            4: self.h4,
            5: self.h5,
            6: self.h6,
            7: self.h7,
            8: self.h8,
            9: self.h9,
            10: self.h10,
            11: self.h11,
            12: self.h12,
        }

        if level not in level_headings:  # pragma: no cover
            raise ValueError(
                f"The 'level' value must be in {list(level_headings.keys())} but is {level}."
            )

        return level_headings[level](*args, **kwargs)

    def code_block(self, language="", indent=0):
        """Add a code-block directive to the internal buffer."""
        self.newline()
        self.add([f".. code-block:: {language}"], indent)
        self.newline()

    def exception(self, exception=None, indent=0, width=None):
        """Format and add an exception to the internal buffer."""
        self.newline()
        if exception is None:
            self.add("")
        else:
            # Clean ANSI escape sequences (console colors)
            exception = [self._ANSI_re_compiled.sub("", i) for i in exception.split("\n")]

            if width is not None:
                exception = [
                    j
                    for i in exception
                    for j in textwrap.fill(
                        i,
                        width=width,
                        break_on_hyphens=False,
                        break_long_words=True,
                    ).split("\n")
                ]
            self.add(exception, indent)
        self.newline()

    def list(self, elements, width=None, indent=0, bullet="*"):
        """Add list elements to the internal buffer."""
        for elem in elements:
            elem = f"{bullet} {elem}"
            if width is not None:
                elem = textwrap.fill(
                    elem,
                    width=width,
                    break_on_hyphens=False,
                    break_long_words=True,
                    expand_tabs=False,
                    initial_indent=" " * indent,
                    replace_whitespace=False,
                    subsequent_indent=" " * (indent + 2),
                )
            self.add(elem)
