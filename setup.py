from __future__ import annotations

from setuptools import Extension, setup

ext_modules = [Extension("chgnet.graph.cygraph", ["chgnet/graph/cygraph.pyx"])]

setup(ext_modules=ext_modules, setup_requires=["Cython"])
