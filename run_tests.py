#!/usr/bin/env python
import argparse
import os
import sys
import unittest
import importlib

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--verbosity', type=int, default=2)
    args = parser.parse_args()
    return args

if __name__ == "__main__" and __package__ is None:
    ARGS = get_arguments()

    # __file__ = "run_tests.py"
    SCRIPT_PATH = os.path.realpath(__file__)   # "/some/path/to/mypackage/run_tests.py"
    PACKAGE_DIR = os.path.dirname(SCRIPT_PATH) # "/some/path/to/mypackage"
    PARENT_PACKAGE_DIR, PACKAGE_NAME = os.path.split(PACKAGE_DIR) #("/some/path/to", "mypackage")

    #append the directory above the package-directory to PYTHONPATH
    sys.path.append(PARENT_PACKAGE_DIR)

    hdf5handler = importlib.import_module(PACKAGE_NAME)
    __package__ = PACKAGE_NAME

    from hdf5handler.tests import tests
    from hdf5handler.colored import ColoredTextTestRunner

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(tests)

    runner = ColoredTextTestRunner(verbosity=2)
    results = runner.run(suite)

    if (len(results.failures) or len(results.errors)) > 0:
        exit(1)
    else:
        exit(0)




