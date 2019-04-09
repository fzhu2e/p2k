#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import sys
import os
import re

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

meta_file = open("p2k/__init__.py").read()
metadata = dict(re.findall("__([a-z]+)__\s*=\s*'([^']+)'", meta_file))

setup(
    name='p2k',
    version=metadata['version'],
    description="A package to make life easier with PAGES2k dataset and stuff.",
    long_description=readme,
    author=metadata['author'],
    author_email=metadata['email'],
    url='https://github.com/fzhu2e/p2k',
    packages=find_packages(),
    include_package_data=True,
    license="MIT license",
    zip_safe=False,
    keywords='p2k',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
