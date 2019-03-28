##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import subprocess

import setuptools.command.develop 
import setuptools.command.install 

from setuptools import setup, find_packages


cwd = os.path.dirname(os.path.abspath(__file__))

version = '0.5.1'
try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
    version += '+' + sha[:7]
except Exception:
    pass

def create_version_file():
    global version, cwd
    print('-- Building version ' + version)
    version_path = os.path.join(cwd, 'encoding', 'version.py')
    with open(version_path, 'w') as f:
        f.write('"""This is encoding version file."""\n')
        f.write("__version__ = '{}'\n".format(version))

# run test scrip after installation
class install(setuptools.command.install.install):
    def run(self):
        create_version_file()
        setuptools.command.install.install.run(self)

class develop(setuptools.command.develop.develop):
    def run(self):
        create_version_file()
        setuptools.command.develop.develop.run(self)

try:
    import pypandoc
    readme = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    readme = open('README.md').read()

requirements = [
    'numpy',
    'tqdm',
    'nose',
    'torch>=1.0.0',
    'cffi>=1.0.0',
]

requirements = [
    'numpy',
    'tqdm',
    'nose',
    'torch>=1.0.0',
    'Pillow',
    'scipy',
    'requests',
]

setup(
    name="fast-fcn",
    version=version,
    author="Huikai Wu",
    author_email="huikai.wu@nlpr.ia.ac.cn",
    url="https://github.com/wuhuikai/FastFCN",
    description="PyTorch Encoding Package",
    long_description=readme,
    license='MIT',
    install_requires=requirements,
    packages=find_packages(exclude=["images", "experiments"]),
    package_data={'encoding': [
        'LICENSE',
        'lib/cpu/*.h',
        'lib/cpu/*.cpp',
        'lib/gpu/*.h',
        'lib/gpu/*.cpp',
        'lib/gpu/*.cu',
    ]},
    cmdclass={
        'install': install,
        'develop': develop,
    },
)
