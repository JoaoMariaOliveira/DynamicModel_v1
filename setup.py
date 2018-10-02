import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

# OpenMP: http://docs.cython.org/en/latest/src/userguide/parallelism.html#compiling
ext_modules = [
    Extension(
        'expenditure_aux',
        ['expenditure_aux.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

setup(
    ext_modules = cythonize(ext_modules, language='c'),
    include_dirs=[numpy.get_include()]
)
