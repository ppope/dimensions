# import os
# from os.path import join
#
#
# def configuration(parent_package='', top_path=None):
#     import numpy
#     from numpy.distutils.misc_util import Configuration
#
#     config = Configuration('utils', parent_package, top_path)
#
#     libraries = []
#     if os.name == 'posix':
#         libraries.append('m')
#
#     config.add_extension('graph_shortest_path',
#                          sources=['graph_shortest_path.pyx'],
#                          include_dirs=[numpy.get_include()])
#
#     return config
#
#
# if __name__ == '__main__':
#     from numpy.distutils.core import setup
#     setup(**configuration(top_path='').todict())

import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='gsp',
    ext_modules=cythonize("graph_shortest_path.pyx"),
    include_dirs=[numpy.get_include()],
    # zip_safe=False,
)
