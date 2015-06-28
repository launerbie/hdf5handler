#!/usr/bin/env python
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

class TestCommand(Command):
    description = "run all tests"
    user_options = [] # distutils complains if this is not here.

    def __init__(self, *args):
        self.args = args[0]
        Command.__init__(self, *args)

    def initialize_options(self):  # distutils wants this
        pass

    def finalize_options(self):    # this too
        pass

    def run(self):
        from hdf5handler.tests import runtests
        runtests.run_all_tests()


setup(
    name = "hdf5handler",
    version = "0.0.1",
    packages = ['hdf5handler'],
    cmdclass = {'clean': CleanCommand,
                'test': TestCommand,}
)

