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
MAINTAINER = "Rick van Veen"
MAINTAINER_EMAIL = "r.van.veen133@gmail.com"
URL = "https://github.com/rickvanveen/sklvq"
LICENSE = "The 3-Clause BSD License"
DOWNLOAD_URL = "https://github.com/rickvanveen/sklvq"
VERSION = __version__
INSTALL_REQUIRES = ["numpy>=1.14.0", "scipy>=1.1.0", "scikit-learn>=0.23.2"]
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
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
]
EXTRAS_REQUIRE = {
    "tests": ["pytest", "pytest-cov", "coverage", "pandas"],
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
    long_description_content_type="text/x-rst",
    zip_safe=False,
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires='>=3.6'
)
