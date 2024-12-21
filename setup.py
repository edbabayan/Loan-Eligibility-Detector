import io
import os
from pathlib import Path

from setuptools import find_packages, setup


name = "loan_model"
description = "Loan Eligibility Model"
url = "https://github.com/edbabayan/Loan-Eligibility-Detector"
email = "ebabayan86@gmail.com"
author = "Eduard Babayan"
required_python = ">=3.10.0"

pwd = os.path.abspath(os.path.dirname(__file__))


def list_reqs(fname="requirements.txt"):
    with io.open(os.path.join(pwd, fname), encoding="utf-8") as f:
        return f.read().splitlines()


try:
    with io.open(os.path.join(pwd, "README.md"), encoding="utf-8") as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = description

root_dir = Path(__file__).resolve().parent
about = {}

with open(root_dir / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version

setup(
    name=name,
    version=about["__version__"],
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=author,
    author_email=email,
    python_requires=required_python,
    url=url,
    packages=find_packages(exclude='tests'),
    package_data={"loan_model": ["VERSION"]},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license="MIT",
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ]
)
