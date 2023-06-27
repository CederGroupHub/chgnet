from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = cythonize([Extension("chgnet.graph.cygraph", ["chgnet/graph/cygraph.pyx"])])

setup(
    ext_modules = extensions
)