from __future__ import annotations

import numpy as np
from setuptools import Extension, setup

ext_modules = [Extension("chgnet.graph.cygraph", ["chgnet/graph/cygraph.pyx"])]

setup(
    ext_modules=ext_modules, setup_requires=["Cython"], include_dirs=[np.get_include()]
)
