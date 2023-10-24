# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path
import pathlib

import pkg_resources

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# requirements
with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name='semireward',
    version='0.1.0',
    description='SemiReward: A General Reward Model for Semi-supervised Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Westlake-AI/SemiReward/',
    author='Siyuan Li and Weiyang Jin and Zedong Wang and Fang Wu and Zicheng Liu and Cheng Tan and Stan Z. Li',
    author_email='lisiyuan@westlake.edu.cn, wayneyjin@gmail.com',

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='pytorch, semi-supervised-learning, reward modeling',
    packages=find_packages(exclude=['preprocess', 'saved_models', 'data', 'config']),
    include_package_data=True,
    # install_requires=['torch >= 1.8', 'torchvision', 'torchaudio', 'transformers', 'timm', 'progress', 'ruamel.yaml', 'scikit-image', 'scikit-learn', 'tensorflow', ''],
    install_requires=install_requires,
    license='Apache License 2.0',
    python_requires='>=3.9',
)
