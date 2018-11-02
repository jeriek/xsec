#! /usr/bin/env python

"""
Setup information to allow installation with pip.
"""
from setuptools import setup  # find_packages


def version():
    with open('VERSION') as version_file:
        return version_file.read().strip()


def readme():
    """
    Define the README text used by PyPI to build the package homepage.
    """
    # Set to use the text in README.md
    with open('README.md') as readme_file:
        return readme_file.read()


# Add the package metadata and specify which files to include in the
# distribution. Setting packages=find_packages() would include code in
# 'xsec.data', but we only want to include the init file (to ensure the
# folder is created) and not transform.py, which comes separately
# (together with the data files). So this is explicitly specified in
# the py_modules keyword.
setup(name='xsec',
      version=version(),
      description='Cross-section evaluation code',
      long_description=readme(),
      classifiers=[
          'Development Status :: 1 - Planning',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering:: Physics',
      ],
      keywords='supersymmetry cross-sections machine-learning',
      url='https://github.com/jeriek/xsec',
      maintainer='Jeriek Van den Abeele',
      maintainer_email='jeriekvda@fys.uio.no',
      license='GPLv3',
      packages=['xsec'],
      py_modules=['xsec.gprocs.__init__'],
      install_requires=[
          'numpy>=1.14',
          'joblib>=0.12.2',
          'pyslha>=3.2.0'
      ],
      include_package_data=False,
      scripts=['scripts/xsec-download-gprocs', 'scripts/xsec-test', 'scripts/xsec'],
      zip_safe=False)
