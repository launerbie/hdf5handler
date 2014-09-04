#!/usr/bin/env python
import argparse
import os
import sys

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--verbosity', type=int, default=2)
    args = parser.parse_args()
    return args

if __name__ == "__main__" and __package__ == "hdf5handler":
    from .tests import tests
    tests.run()

elif __name__ == "__main__" and __package__ is None:
    ARGS = get_arguments()
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    import hdf5handler
    __package__ = "hdf5handler"

    from .tests import tests
    results = tests.run(verbositylvl=ARGS.verbosity)

    if (len(results.failures) or len(results.errors)) > 0:
        exit(1)
    else:
        exit(0)


