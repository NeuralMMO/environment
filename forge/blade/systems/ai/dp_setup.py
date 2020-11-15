from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
   ext_modules=cythonize("dynamic_programming.pyx"),
   include_dirs=[numpy.get_include()]
)

setup(
   ext_modules=[
      Extension("dynamic_programming", ["dynamic_programming.c"],
                include_dirs=[numpy.get_include()]),
   ],
)

# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()
