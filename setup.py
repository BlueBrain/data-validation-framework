"""Setup for the data-validation-framework package."""
from setuptools import find_packages
from setuptools import setup

# Read the contents of the README file
with open("README.md", encoding="utf-8") as f:
    README = f.read()

reqs = [
    "luigi>=3.1",
    "luigi-tools>=0.0.18",
    "pandas",
    "rst2pdf",
    "sphinx>=3,<5",
    "tqdm",
]
doc_reqs = [
    "m2r2",
    "mistune<2",
    "sphinx-bluebrain-theme",
]
test_reqs = [
    "diff_pdf_visually>=1.6.2",
    "pause",
    "pytest",
    "pytest-cov",
    "pytest-html",
    "rst2pdf>=0.99",
]

setup(
    name="data-validation-framework",
    author="Blue Brain Project, EPFL",
    description="Simple framework to create data validation workflows.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://data-validation-framework.readthedocs.io/en/latest/",
    project_urls={
        "Tracker": "https://github.com/BlueBrain/data-validation-framework/issues",
        "Source": "https://github.com/BlueBrain/data-validation-framework",
    },
    license="Apache-2.0",
    packages=find_packages(include=["data_validation_framework"]),
    python_requires=">=3.8",
    use_scm_version=True,
    setup_requires=[
        "setuptools_scm",
    ],
    install_requires=reqs,
    extras_require={
        "docs": doc_reqs,
        "test": test_reqs,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
