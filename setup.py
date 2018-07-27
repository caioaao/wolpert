#!/usr/bin/env python

import sys
import os

from setuptools import setup, find_packages
from setuptools.command.install import install

VERSION = "0.1b1"


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}" \
                   .format(tag, VERSION)
            sys.exit(info)

with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()

setup(name="wolpert",
      version=VERSION,
      description="Stacked generalization framework",
      long_description=LONG_DESCRIPTION,
      url="https://github.com/caioaao/wolpert",
      author="Caio Oliveira",
      license="new BSD",
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved',
                   'Programming Language :: Python',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6'],
      package_dir={'wolpert': 'wolpert'},
      packages=find_packages(include=['wolpert', 'wolpert.*'],
                             exclude=['*.tests', '*.tests.*']),
      install_requires=["numpy>=0.14.2", "scikit-learn>=0.19.1"],
      cmdclass={
          'verify': VerifyVersionCommand,
      })
