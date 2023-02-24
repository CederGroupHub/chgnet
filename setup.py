# -*- coding: utf-8 -*-
"""
Created on Feb 22 2023
@author: Bowen Deng
"""
import re
import os
from setuptools import setup, find_packages, find_namespace_packages

module_dir = os.path.dirname(os.path.abspath(__file__))

with open("chgnet/__init__.py", encoding="utf-8") as fd:
    try:
        lines = ""
        for item in fd.readlines():
            item = item
            lines += item + "\n"
    except Exception as exc:
        raise Exception(f"Caught exception {exc}")
version = re.search('__version__ = "(.*)"', lines).group(1)

if __name__ == "__main__":
    setup(
        name="chgnet",
        version=version,
        description="Pretrained Neural Network Potential for Charge-informed Molecular Dynamics",
        long_description=open(os.path.join(module_dir, "README.md")).read(),
        url="https://github.com/BowenD-UCB/chgnet",
        author=["Bowen Deng"],
        author_email=["bowendeng@berkeley.edu"],
        license="MIT",
        packages=find_namespace_packages(),
        package_data={"chgnet": ["chgnet/pretrained/*"]},
        include_package_data=True,
        include_dirs=["pretrained"],
        zip_safe=False,
        install_requires=[],
        extras_require={},
        classifiers=[],
        test_suite="",
        tests_require=[],
        scripts=[],
    )
