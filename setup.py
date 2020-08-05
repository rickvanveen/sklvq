#! /usr/bin/env python
import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join("sklvq", "_version.py")
with open(ver_file) as f:
    exec(f.read())

DISTNAME = "sklvq"
DESCRIPTION = "A collection of Learning Vector Quantization algorithms compatible with scikit-learn"
with codecs.open("README.rst", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "R. van Veen"
MAINTAINER_EMAIL = "r.van.veen133@gmail.com"
URL = "https://github.com/rickvanveen/sklvq"
LICENSE = "GNU Affero General Public License v3.0"
DOWNLOAD_URL = "https://github.com/rickvanveen/sklvq"
VERSION = __version__
INSTALL_REQUIRES = ["numpy", "scipy", "scikit-learn>=0.23.1"]
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
]
EXTRAS_REQUIRE = {
    "tests": ["pytest", "pytest-cov", "pandas"],
    "docs": ["sphinx", "sphinx-gallery", "sphinx_rtd_theme", "numpydoc", "matplotlib", "pillow"],
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
