#!/usr/bin/env python

from setuptools import setup

import os
from setuptools import setup, Command

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info \
                  ./hdf5handler/__pycache__')


setup(
    name = "hdf5handler",
    version = "0.0.1",
    packages = ['hdf5handler'],
    cmdclass = {'clean': CleanCommand}
)

