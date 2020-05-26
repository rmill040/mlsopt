#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import dirname, join
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
import versioneer


def list_reqs(fname="requirements.txt"):
    """Gather requirements from the requirements.txt file.
    """
    return open(fname).read().splitlines()


def read_file(fname="README.md"):
    """Get contents of file from the module's directory.
    """
    return open(join(dirname(__file__), fname), encoding='utf-8').read()


class BuildExt(build_ext):
    """build_ext command for use when numpy headers are needed.
    SEE tutorial: https://stackoverflow.com/questions/2379898
    SEE fix: https://stackoverflow.com/questions/19919905
    """
    def finalize_options(self):
        build_ext.finalize_options(self)
        import numpy
        self.include_dirs.append(numpy.get_include())


# Package meta-data
MODULE_NAME          = "mlsopt"
AUTHORS              = ",".join(["Robert Milletich", "Anthony Asmar"])
AUTHOR_EMAIL         = "rmill040@gmail.com"
KEYWORDS             = "stochastic optimization machine learning"
SHORT_DESCRIPTION    = "Stochastic optimization of machine learning pipelines"
LONG_DESCRIPTION     = read_file()
CONTENT_TYPE         = "text/markdown"
MODULE_URL           = "https://github.com/rmill040/mlsopt"
PLATFORMS            = "any"
TEST_SUITE           = "pytest"
SETUP_REQS           = ["numpy", "cython"]
INSTALL_REQS         = list_reqs()
PACKAGES             = find_packages(exclude=['tests'])
CMDCLASS             = {"build_ext": BuildExt}
MIN_PYTHON_VERSION   = ">=3.6.*"
VERSION              = versioneer.get_version()
PACKAGE_DATA         = {}
# setup(
#     package_data = {
#         'my_package': ['*.pxd'],
#         'my_package/sub_package': ['*.pxd'],
#     },
#     ...
# )
INCLUDE_PACKAGE_DATA = True
EXTRAS_REQUIRE       = {}
SCRIPTS              = []
LICENSE              = "MIT"
ZIP_SAFE             = False
CLASSIFIERS          = ['Programming Language :: Python :: 3',
                        'Development Status :: 3 - Alpha',
                        'Natural Language :: English',
                        'Intended Audience :: Developers',
                        'Intended Audience :: Education',
                        'Intended Audience :: Science/Research',
                        'License :: OSI Approved :: MIT License',
                        'Operating System :: OS Independent',
                        'Topic :: Scientific/Engineering :: Artificial Intelligence',
                        'Topic :: Scientific/Engineering',
                        'Topic :: Software Development']

# Define Cython extensions
EXTENSIONS = []
# EXTENSIONS = [Extension('mlsopt.base.optimizers',
#                         sources=['mlsopt/base/optimizers.pyx']),
#               Extension('mlsopt.base.samplers',
#                         sources=['mlsopt/base/samplers.pyx'])
#         ]

# Define Cython compiler directives
COMPILER_DIRECTIVES = {
    'boundscheck' : False,
    'wraparound'  : False,
}

for e in EXTENSIONS:
    e.cython_directives = COMPILER_DIRECTIVES

# Run setup
setup(
    name                          = MODULE_NAME,
    url                           = MODULE_URL,
    author                        = AUTHORS,
    author_email                  = AUTHOR_EMAIL,
    python_requires               = MIN_PYTHON_VERSION,
    version                       = VERSION,
    cmdclass                      = CMDCLASS,
    ext_modules                   = EXTENSIONS,
    test_suite                    = TEST_SUITE,
    setup_requires                = SETUP_REQS,
    keywords                      = KEYWORDS,
    description                   = SHORT_DESCRIPTION,
    long_description              = LONG_DESCRIPTION,
    long_description_content_type = CONTENT_TYPE,
    packages                      = PACKAGES,
    platforms                     = PLATFORMS,
    scripts                       = SCRIPTS,
    package_data                  = PACKAGE_DATA,
    include_package_data          = INCLUDE_PACKAGE_DATA,
    install_requires              = INSTALL_REQS,
    extras_require                = EXTRAS_REQUIRE,
    license                       = LICENSE,
    classifiers                   = CLASSIFIERS,
    zip_safe                      = ZIP_SAFE
)