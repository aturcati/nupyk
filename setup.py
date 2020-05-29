# -*- coding: utf-8 -*-

from distutils.core import setup
from setuptools import find_packages
import os


# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''

setup(
    # Name of the package
    name="nupyk",

    # Packages to include into the distribution
    packages=find_packages('.'),

    # Start with a small number and increase it with every change you make
    # https://semver.org
    version='0.0.1',

    # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    # For example: MIT
    license='gpl-3.0',

    # Short description of your library
    description='Machine Learning Tools to analyse Spectral Energy Distributions of Blazars',

    # Long description of your library
    long_description = long_description,
    long_description_context_type = 'text/markdown',

    # Your name
    author='Andrea Turcati',

    # Your email
    author_email='aturcati@gmail.com',

    # Either the link to your github or to your website
    url='https://github.com/aturcati',

    # Link from which the project can be downloaded
    download_url='https://github.com/aturcati/nupyk',

    # List of keyword arguments
    keywords=["Science", "Astrophysics", "Machine Learning", "Blazar", "AGN"],

    # List of packages to install with this one
    install_requires=[
        'numpy>=1.18',
        'pandas>=1.0',
        'scipy>=1.4',
        'tqdm>=4.43',
        'xgboost>=1.0.2',
        'seaborn>=0.10',
        'pandas-profiling>=2.8.0'
    ],

    # https://pypi.org/classifiers/
    classifiers=[
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Operating System :: Unix",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
    ]
)
