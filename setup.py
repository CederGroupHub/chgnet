from __future__ import annotations

from setuptools import Extension, setup

ext_modules = [Extension("chgnet.graph.cygraph", ["chgnet/graph/cygraph.pyx"])]


def lazy_numpy_include() -> str:
    """Get the numpy include directory lazily."""
    import numpy as np

    return np.get_include()


setup(
    ext_modules=ext_modules,
    setup_requires=["Cython"],
    include_dirs=[lazy_numpy_include()],
)
