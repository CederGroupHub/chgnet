from __future__ import annotations

from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = cythonize(
    [Extension("chgnet.graph.cygraph", ["chgnet/graph/cygraph.pyx"])]
)

setup(ext_modules=extensions)
