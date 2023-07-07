"""Setup for the data-validation-framework package."""
from pathlib import Path

from setuptools import find_namespace_packages
from setuptools import setup

reqs = [
    "luigi>=3.1",
    "luigi-tools>=0.0.18",
    "numpy>=1.21",
    "pandas>=1.3",
    "rst2pdf>=0.99",
    "sphinx>=4,<8",
    "tqdm>=4.40",
]
doc_reqs = [
    "m2r2",
    "sphinx",
    "sphinx-bluebrain-theme",
]
test_reqs = [
    "diff_pdf_visually>=1.6.2",
    "pause>=0.2",
    "pytest>=7",
    "pytest-cov>=3",
    "pytest-html>=3.1",
]

setup(
    name="data-validation-framework",
    author="Blue Brain Project, EPFL",
    description="Simple framework to create data validation workflows.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://data-validation-framework.readthedocs.io",
    project_urls={
        "Tracker": "https://github.com/BlueBrain/data-validation-framework/issues",
        "Source": "https://github.com/BlueBrain/data-validation-framework",
    },
    license="Apache License 2.0",
    packages=find_namespace_packages(include=["data_validation_framework*"]),
    python_requires=">=3.8",
    install_requires=reqs,
    extras_require={
        "docs": doc_reqs,
        "test": test_reqs,
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
