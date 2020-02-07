#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
from os.path import join
from pathlib import Path
from setuptools import find_packages, setup
import versioneer

# Package meta-data
NAME            = "mlsopt"
DESCRIPTION     = "Stochastic optimization of machine learning pipelines"
URL             = "https://github.com/rmill040/mlsopt"
EMAIL           = "rmill040@gmail.com"
AUTHOR          = "Robert Milletich"
REQUIRES_PYTHON = ">=3.6.0"

# Requirements for project
def list_reqs(fname="requirements.txt"):
    with open(fname) as fd:
        return fd.read().splitlines()

# Import the README and use it as the long-description
here = Path(__file__).resolve().parent
try:
    with io.open(join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Define scripts
scripts = []

# Run setup
setup(
    name                          = NAME,
    version                       = versioneer.get_version(),
    cmdclass                      = versioneer.get_cmdclass(),
    description                   = DESCRIPTION,
    long_description              = long_description,
    long_description_content_type = 'text/markdown',
    author                        = AUTHOR,
    author_email                  = EMAIL,
    python_requires               = REQUIRES_PYTHON,
    url                           = URL,
    packages                      = find_packages(exclude=['tests']),
    scripts                       = scripts,
    package_data                  = {},
    install_requires              = list_reqs(),
    extras_require                = {},
    include_package_data          = True,
    license                       = 'MIT',
    classifiers                   = [
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython'
    ],
)