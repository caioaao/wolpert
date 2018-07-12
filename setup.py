#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name="wolpert",
      description="Stacked generalization framework",
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
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6'],
      package_dir={'wolpert': 'wolpert'},
      packages=find_packages(include=['wolpert', 'wolpert.*'],
                             exclude=['*.tests', '*.tests.*']),
      version="0.0.1",
      install_requires=["numpy>=0.14.2", "scikit-learn>=0.19.1"],
      include_package_data=True,
      zip_safe=False)
