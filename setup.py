#!/usr/bin/env python

import os

from setuptools import find_packages, setup

install_requires = [
    "torch>=1.11.0",
    "matplotlib",
    "numpy",  # Due to pandas incompatibility
    "scipy",
    "scikit-learn",
    "torchdyn>=1.0.6",
    "pot",
    "torchdiffeq",
    "absl-py",
    "pandas>=2.2.2",
]

version_py = os.path.join(os.path.dirname(__file__), "torchlfm", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()
readme = open("README.md", encoding="utf8").read()
setup(
    name="torchlfm",
    version=version,
    description="Lagrangian Flow Matching: A Least-Action Framework for Principled Path Design.",
    author="Shukai Du, Junzhe Zhang, Yiming Li",
    author_email="jzhan403@syr.edu, sdu113@syr.edu",
    url="https://github.com/junzhez/lagrangian-flow-matching",
    install_requires=install_requires,
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
    extras_require={"forest-flow": ["xgboost", "scikit-learn", "ForestDiffusion"]},
)
