"""Created on Feb 22 2023
@author: Bowen Deng.
"""
import os

from setuptools import find_namespace_packages, setup

module_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    setup(
        name="chgnet",
        version="0.0.1",
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
        install_requires=[
            "numpy~=1.21.6",
            "torch~=1.11.0",
            "pymatgen~=2022.4.19",
            "ase==3.22.0",
        ],
        extras_require={"test": ["pytest", "pytest-cov"]},
        classifiers=[],
        test_suite="",
        tests_require=[],
        scripts=[],
    )
