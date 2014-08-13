#!/usr/bin/env python

import unittest
import argparse
import os
import h5py
import numpy

from hdf5handler import HDF5Handler
from colored import ColoredTextTestRunner


class test_Base(unittest.TestCase):
    def setUp(self):
        self.filename = 'test.hdf5'

    def tearDown(self):
        try:
            os.remove(self.filename)
        except OSError:
            pass


class test_file_group_dataset_creation(test_Base):
    def test_open_hdf5(self):
        with HDF5Handler(self.filename) as handler:
            self.assertTrue(os.path.exists(self.filename))

    def test_group_creation(self):
        with HDF5Handler(self.filename) as handler:
            handler.put(1, 'grp/dset')
            self.assertTrue(isinstance(handler.file['grp'], h5py.Group))

    def test_hdf5file_dataset_creation(self):
        with HDF5Handler(self.filename) as handler:
            handler.put(1, 'dset')
            self.assertTrue(isinstance(handler.file['dset'], h5py.Dataset))

    def test_group_and_dataset_creation(self):
        with HDF5Handler(self.filename) as handler:
            handler.put(1, 'grp/dset')
            self.assertTrue(isinstance(handler.file['grp/dset'], h5py.Dataset))
            self.assertTrue(isinstance(handler.file['grp']['dset'],
                                       h5py.Dataset))

    def test_group_creation_after_closing(self):
        with HDF5Handler(self.filename) as handler:
            handler.put(1, 'grp/dset')

        f = h5py.File(self.filename)
        self.assertTrue(isinstance(f['grp'], h5py.Group))

    def test_hdf5file_dataset_creation_after_closing(self):
        with HDF5Handler(self.filename) as handler:
            handler.put(1, 'dset')
            self.assertTrue(isinstance(handler.file['dset'], h5py.Dataset))

        f = h5py.File(self.filename)
        self.assertTrue(isinstance(f['dset'], h5py.Dataset))

    def test_group_and_dataset_creation_after_closing(self):
        with HDF5Handler(self.filename) as handler:
            handler.put(1, 'grp/dset')

        f = h5py.File(self.filename)
        self.assertTrue(isinstance(f['grp/dset'], h5py.Dataset))
        self.assertTrue(isinstance(f['grp']['dset'], h5py.Dataset))

    def test_creation_multiple_datasets(self):
        with HDF5Handler(self.filename) as handler:
            handler.put(1, 'testA')
            handler.put(1, 'testB')
            handler.put(1, 'testC')
            self.assertTrue(isinstance(handler.file['testA'], h5py.Dataset))
            self.assertTrue(isinstance(handler.file['testB'], h5py.Dataset))
            self.assertTrue(isinstance(handler.file['testC'], h5py.Dataset))
            self.assertEqual(3, len(handler.file.keys()))

    def test_creation_multiple_datasets_after_closing(self):
        with HDF5Handler(self.filename) as handler:
            handler.put(1, 'testA')
            handler.put(1, 'testB')
            handler.put(1, 'testC')

        f = h5py.File(self.filename)
        self.assertTrue(isinstance(f['testA'], h5py.Dataset))
        self.assertTrue(isinstance(f['testB'], h5py.Dataset))
        self.assertTrue(isinstance(f['testC'], h5py.Dataset))
        self.assertEqual(3, len(f.keys()))


class test_python_scalars(test_Base):
    def test_ints(self):
        with HDF5Handler(self.filename) as handler:
            handler.put(1, 'testint')
            handler.put(1, 'testint')
            handler.put(1, 'testint')

        f = h5py.File(self.filename)
        self.assertEqual(3, f['testint'].value.sum())

    def test_floats(self):
        with HDF5Handler(self.filename) as handler:
            handler.put(1.1, 'testfloat')
            handler.put(1.1, 'testfloat')
            handler.put(1.1, 'testfloat')

        f = h5py.File(self.filename)
        self.assertAlmostEqual(3.3, f['testfloat'].value.sum(), places=7)

    def test_bools(self):
        with HDF5Handler(self.filename) as handler:
            handler.put(True, 'testbool')
            handler.put(True, 'testbool')
            handler.put(False, 'testbool')

        f = h5py.File(self.filename)
        self.assertEqual(2, f['testbool'].value.sum())


class test_python_lists(test_Base):
    def test_list_bool(self):
        with HDF5Handler(self.filename) as handler:
            handler.put([True, True, True], 'list')
            handler.put([True, True, True], 'list')
            handler.put([True, True, False], 'list')

        f = h5py.File(self.filename)
        self.assertEqual(8, f['list'].value.sum())

    def test_list_integer(self):
        with HDF5Handler(self.filename) as handler:
            handler.put([1, 1, 1], 'list')
            handler.put([1, 1, 1], 'list')
            handler.put([1, 1, 0], 'list')

        f = h5py.File(self.filename)
        self.assertEqual(8, f['list'].value.sum())

    def test_list_float(self):
        with HDF5Handler(self.filename) as handler:
            handler.put([1.1, 1.1, 1.1], 'list')
            handler.put([1.1, 1.1, 1.1], 'list')
            handler.put([1.1, 1.1, 1.1], 'list')

        f = h5py.File(self.filename)
        self.assertAlmostEqual(9.9, f['list'].value.sum(), places=7)

    def test_list_nested_bool(self):
        with HDF5Handler(self.filename) as handler:
            handler.put([[True], [True], [True]], 'list')
            handler.put([[True], [True], [True]], 'list')
            handler.put([[True], [True], [False]], 'list')

        f = h5py.File(self.filename)
        self.assertEqual(8, f['list'].value.sum())

    def test_list_nested_integer(self):
        with HDF5Handler(self.filename) as handler:
            handler.put([[1], [1], [1]], 'list')
            handler.put([[1], [1], [1]], 'list')
            handler.put([[1], [1], [0]], 'list')

        f = h5py.File(self.filename)
        self.assertEqual(8, f['list'].value.sum())

    def test_list_nested_float(self):
        with HDF5Handler(self.filename) as handler:
            handler.put([[1.1], [1.1], [1.1]], 'list')
            handler.put([[1.1], [1.1], [1.1]], 'list')
            handler.put([[1.1], [1.1], [0.1]], 'list')

        f = h5py.File(self.filename)
        self.assertAlmostEqual(8.9, f['list'].value.sum(), places=7)


class test_python_tuples(test_Base):
    def test_tuple_bool(self):
        with HDF5Handler(self.filename) as handler:
            handler.put((True, True, True), 'tup')
            handler.put((True, True, True), 'tup')
            handler.put((True, True, False), 'tup')

        f = h5py.File(self.filename)
        self.assertEqual(8, f['tup'].value.sum())

    def test_tuple_list(self):
        with HDF5Handler(self.filename) as handler:
            handler.put((1, 1, 1), 'tup')
            handler.put((1, 1, 1), 'tup')
            handler.put((1, 1, 0), 'tup')

        f = h5py.File(self.filename)
        self.assertEqual(8, f['tup'].value.sum())

    def test_tuple_float(self):
        with HDF5Handler(self.filename) as handler:
            handler.put((1.1, 1.1, 1.1), 'tup')
            handler.put((1.1, 1.1, 1.1), 'tup')
            handler.put((1.1, 1.1, 1.1), 'tup')

        f = h5py.File(self.filename)
        self.assertAlmostEqual(9.9, f['tup'].value.sum(), places=7)

    def test_tuple_nested_bool(self):
        with HDF5Handler(self.filename) as handler:
            handler.put(((True,), (True,), (True,)), 'tup')
            handler.put(((True,), (True,), (True,)), 'tup')
            handler.put(((True,), (True,), (False,)), 'tup')

        f = h5py.File(self.filename)
        self.assertEqual(8, f['tup'].value.sum())

    def test_tuple_nested_integer(self):
        with HDF5Handler(self.filename) as handler:
            handler.put(((1,), (1,), (1,)), 'tup')
            handler.put(((1,), (1,), (1,)), 'tup')
            handler.put(((1,), (1,), (0,)), 'tup')

        f = h5py.File(self.filename)
        self.assertEqual(8, f['tup'].value.sum())

    def test_tuple_nested_float(self):
        with HDF5Handler(self.filename) as handler:
            handler.put(((1.1,), (1.1,), (1.1,)), 'tup')
            handler.put(((1.1,), (1.1,), (1.1,)), 'tup')
            handler.put(((1.1,), (1.1,), (0.1,)), 'tup')

        f = h5py.File(self.filename)
        self.assertAlmostEqual(8.9, f['tup'].value.sum(), places=7)


class test_ndarrays(unittest.TestCase):
    def setUp(self):
        self.filename = 'test.hdf5'

        self.ints1d = numpy.ones(12345*2)
        self.floats1d = numpy.linspace(0, 4123, 10000*2)
        self.ints = numpy.ones(12345*2).reshape(12345, 2)
        self.floats = numpy.linspace(0, 4123, 10000*2).reshape(10000, 2)

        self.sumints1d = numpy.sum(self.ints1d)
        self.sumfloats1d = numpy.sum(self.floats1d)
        self.sumints = numpy.sum(self.ints)
        self.sumfloats = numpy.sum(self.floats)

        self.kwargs = dict(chunksize=1000, blockfactor=100) #choose wisely!

    def tearDown(self):
        try:
            os.remove(self.filename)
        except OSError:
            pass


    def test_flushbuffers(self):
        with HDF5Handler(self.filename) as handler:
            for row in self.ints:
                handler.put(row, 'test')

        f = h5py.File(self.filename)
        self.assertEqual(self.sumints, f['test'].value.sum())

    def test_trimming(self):
        with HDF5Handler(self.filename) as handler:
            for row in self.ints:
                handler.put(row, 'test')

        f = h5py.File(self.filename)
        self.assertEqual(self.ints.shape, f['test'].shape)

    def test_flushbuffers_and_trim(self):
        with HDF5Handler(self.filename) as handler:
            for row in self.ints:
                handler.put(row, 'test')

        f = h5py.File(self.filename)
        self.assertEqual(self.sumints, f['test'].value.sum())
        self.assertEqual(self.ints.shape, f['test'].shape)

    def test_shape_scalars(self):
        with HDF5Handler(self.filename) as handler:
            for row in self.ints1d:
                handler.put(row, 'test')

        f = h5py.File(self.filename)
        self.assertEqual(self.ints1d.shape, f['test'].shape)

    def test_shape_arrays(self):
        with HDF5Handler(self.filename) as handler:
            for row in self.ints:
                handler.put(row, 'test')

        f = h5py.File(self.filename)
        self.assertEqual(self.ints.shape, f['test'].shape)


    #####################   Value tests  ####################

    def test_sum_ints_scalar(self):
        with HDF5Handler(self.filename) as handler:
            for element in self.ints1d:
                handler.put(element, 'test')

        f = h5py.File(self.filename)
        self.assertEqual(self.sumints1d, f['test'].value.sum())

    def test_sum_flts_scalar_almostequal6(self):
        with HDF5Handler(self.filename) as handler:
            for element in self.floats1d:
                handler.put(element, 'test')

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats1d, f['test'].value.sum(),
                               places=6)

    def test_sum_flts_scalar_almostequal4(self):
        with HDF5Handler(self.filename) as handler:
            for element in self.floats1d:
                handler.put(element, 'test')

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats1d, f['test'].value.sum(),
                               places=4)

    def test_sum_flts_scalar_almostequal2(self):
        with HDF5Handler(self.filename) as handler:
            for element in self.floats1d:
                handler.put(element, 'test')

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats1d, f['test'].value.sum(),
                               places=2)

    def test_sum_ints_array(self):
        with HDF5Handler(self.filename) as handler:
            for row in self.ints:
                handler.put(row, 'test')

        f = h5py.File(self.filename)
        self.assertEqual(self.sumints, f['test'].value.sum())

    def test_sum_flts_array_almostequal6(self):
        with HDF5Handler(self.filename) as handler:
            for row in self.floats:
                handler.put(row, 'test')

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=6)

    def test_sum_flts_array_almostequal4(self):
        with HDF5Handler(self.filename) as handler:
            for row in self.floats:
                handler.put(row, 'test')

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=4)

    def test_sum_flts_array_almostequal2(self):
        with HDF5Handler(self.filename) as handler:
            for row in self.floats:
                handler.put(row, 'test')

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=2)


class test_prefix(test_Base):
    def test_prefix(self):
        with HDF5Handler(self.filename) as handler:
            handler.prefix = 'prefix/'
            for value in range(10):
                handler.put(value, 'test')

        f = h5py.File(self.filename)
        self.assertEqual(44, f['prefix/test'].value.sum())


class test_shapes(test_Base):
    pass
    #TODO: test for correct shapes, when using nested lists/tuples


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--verbosity', type=int, default=2)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()

    test_cases = [\
                  test_file_group_dataset_creation,
                  test_python_scalars,
                  test_python_lists,
                  test_python_tuples,
                  test_ndarrays,
                  test_prefix,
                 ]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for tc in test_cases:
        tests = loader.loadTestsFromTestCase(tc)
        suite.addTests(tests)

    runner = ColoredTextTestRunner(verbosity=args.verbosity)
    results = runner.run(suite)

    if len(results.failures) or len(results.errors)> 0:
        exit(1)
    else:
        exit(0)


