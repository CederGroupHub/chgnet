from __future__ import annotations

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    """Custom build extension."""

    def finalize_options(self):
        """Override finalize_options to ensure we don't run cythonize when making
        a source distribution.
        """
        import Cython.Build

        self.distribution.ext_modules = Cython.Build.cythonize(
            [Extension("chgnet.graph.cygraph", ["chgnet/graph/cygraph.pyx"])]
        )
        super().finalize_options()


setup(cmdclass={"build_ext": BuildExt})
