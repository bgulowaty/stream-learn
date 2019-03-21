#! /usr/bin/env python
"""Toolbox for streaming data."""
from __future__ import absolute_import

import os
from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('strlearn', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'stream-learn'
DESCRIPTION = 'Python package equipped with a procedures to process data streams using estimators with API compatible with scikit-learn.'
MAINTAINER = 'P. Ksieniewicz'
MAINTAINER_EMAIL = 'pawel.ksieniewicz@pwr.edu.pl'
URL = 'https://github.com/w4k2/stream-learn'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/w4k2/stream-learn'
VERSION = __version__

with open('requirements.txt') as fid:
    INSTALL_REQUIRES = [l.strip() for l in fid.readlines() if l]

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES)
