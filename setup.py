import imp
import sys

from setuptools import find_packages, setup

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

# Read the contents of the README file
with open("README.rst", encoding="utf-8") as f:
    README = f.read()

reqs = [
    "luigi",
    "luigi-tools>=0.0.9",
    "pandas",
    "rst2pdf",
    "sphinx>=3,<4",
    "tqdm",
]
doc_reqs = [
    "sphinx-bluebrain-theme",
]
test_reqs = [
    "pause",
    "diff_pdf_visually>=1.6.2",
    "pytest",
    "pytest-cov",
    "pytest-html",
]

VERSION = imp.load_source("", "data_validation_framework/version.py").VERSION

setup(
    name="data-validation-framework",
    author="bbp-ou-cells",
    author_email="bbp-ou-cells@groupes.epfl.ch",
    version=VERSION,
    description="Simple framework to create data validation workflows.",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://bbpteam.epfl.ch/documentation/projects/data-validation-framework",
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/CELLS/issues",
        "Source": "https://bbpgitlab.epfl.ch/neuromath/data-validation-framework",
    },
    license="BBP-internal-confidential",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.6",
    install_requires=reqs,
    extras_require={
        "docs": doc_reqs,
        "test": test_reqs,
    },
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
