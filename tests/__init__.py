"""Tests of the data-validation-framework package."""
import re
from pathlib import Path


def check_files_exist(directory, pattern_list):
    """Check that a list of file patterns exist in a given directory."""
    existing_files = [str(i.relative_to(directory)) for i in sorted(Path(directory).rglob("*"))]
    try:
        assert len(existing_files) == len(pattern_list)
        for path, pattern in zip(existing_files, pattern_list):
            assert re.match(str(pattern), path)
    except Exception as exc:
        raise RuntimeError(
            f"Error when checking the files.\n\tThe directory is: {directory}\n"
            f"\tThe found files are: {existing_files}\n"
            f"\tThe patterns are: {pattern_list}"
        ) from exc
    return True
