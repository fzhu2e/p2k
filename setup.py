#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import sys
import os

from setuptools import setup, find_packages

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    # TODO: put package requirements here
]

setup_requirements = [
    # TODO(fzhu2e): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='p2k',
    version='0.1.0',
    description="A package to make life easier with PAGES2k dataset and stuff.",
    long_description=readme + '\n\n' + history,
    author="Feng Zhu",
    author_email='fengzhu@usc.edu',
    url='https://github.com/fzhu2e/p2k',
    packages=find_packages(),
    package_data={'p2k': ['f2py_*.so']},
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='p2k',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
