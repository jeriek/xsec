#! /usr/bin/env python

"""
Setup information to allow installation with pip.
"""
from setuptools import setup  # find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='xsec',
      version='0.1.0',
      description='Cross-section evaluation code',
      long_description=readme(),
      classifiers=[
          'Development Status :: 1 - Planning',
          'License :: OSI Approved:: GNU General Public License(GPL)',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering:: Physics',
      ],
      keywords='supersymmetry cross-sections machine-learning',
      url='https://github.com/jeriek/xsec',
      maintainer='Jeriek Van den Abeele',
      maintainer_email='jeriekvda@fys.uio.no',
      license='GPL',
      packages=['xsec'],  # find_packages() includes any code in 'xsec.data'
      py_modules=['xsec.data.__init__'],  # import only init, not transform.py
      install_requires=[
          'numpy>=1.14',
          'joblib>=0.12.2'
      ],
      include_package_data=False,
      zip_safe=False)
