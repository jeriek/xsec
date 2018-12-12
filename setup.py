#! /usr/bin/env python

"""
Setup information to allow installation with pip.
"""
from setuptools import setup


def version():
    with open("VERSION", encoding="utf-8") as version_file:
        return version_file.read().strip()


def readme():
    """
    Define the README text used by PyPI to build the package homepage.
    """
    with open("README.md", encoding="utf-8") as readme_file:
        return readme_file.read()


def requirements():
    with open("requirements.txt") as req_file:
        return req_file.read().splitlines()


# Add the package metadata and specify which files to include in the
# distribution.
setup(
    name="xsec",
    version=version(),
    description="xsec: the cross section evaluation code",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status:: 5 - Production/Stable",
        "License :: OSI Approved :: "
        "GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 2.7",
        #   'Programming Language :: Python :: 3',
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
    ],
    keywords="gaussian processes, supersymmetry, particle, physics, HEP, "
             "cross sections, machine learning",
    url="https://github.com/jeriek/xsec",
    maintainer="Jeriek Van den Abeele",
    maintainer_email="jeriekvda@fys.uio.no",
    license="GPLv3+",
    packages=["xsec"],
    install_requires=requirements(),
    include_package_data=False,
    scripts=[
        "scripts/xsec-download-gprocs",
        "scripts/xsec-test",
        "scripts/xsec",
    ],
)
