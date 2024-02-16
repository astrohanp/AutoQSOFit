#!/usr/bin/env python
import os
import sys
from setuptools import setup, find_packages

# Load the __version__ variable without importing the package already
exec(open('pyqsofit/version.py').read())

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()


setup(
    name='AutoQSOFit',
    version='0.0.1',
    description='Code to automate execution of PyQSOFIT on multiple objects and generating an emission line catalog efficiently',
    author='Rohan Pattnaik, Felix Martinez, and the PyQSOFIT team',
    author_email='rp2503@rit.edu',
    url='https://github.com/astrohanp/AutoQSOFit/',
    # packages=find_packages(),
    package_dir={'pyqsofit': 'pyqsofit'},
      package_data={'pyqsofit': ['fe_uv.txt', 'fe_optical.txt',
                                 'bc03/*.spec.gz',
                                 'pca/Yip_pca_templates/*.fits',
                                 'pca/prior/*.csv',
                                 'indo/*.fits',
                                 'sfddata/*.fits']
                    },
      #data_files=[('bc03',['bc03/*.spec.gz']), ('pca/Yip_pca_templates', ['pca/Yip_pca_templates/*.fits']), ('stddata',['stddata/*.fits'])],
      packages=['pyqsofit'],
      install_requires=install_requires,
      include_package_data=True,
      python_requires='>=3.9',
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
          ],
    
)
