import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        'expenditure_aux',
        ['expenditure_aux.pyx']
    ),
]

setup(
    ext_modules = cythonize(ext_modules, language='c'),
    include_dirs=[numpy.get_include()]
)
