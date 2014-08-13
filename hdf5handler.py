#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import numpy

class HDF5Handler(object):
    """
    The idea is that the HDF5Handler mimics the behaviour of 'open' used as a
    context manager using the 'with' statement:

    >>> with open('myfile.txt','w') as file:
    ...     file.write("This is a line. \n")
    ...     file.write("This is another line. \n")
    ...
    >>>

    which would result in the file 'myfile.txt' containing:
    --------------- myfile.txt ------------------
    This is a line.
    This is another line.
    ---------------------------------------------


    To "write" data with HDF5Handler, simply call it's put() method.

    >>> from hdf5handler import HDF5Handler
    >>> thisdata = [1,2,3]
    >>> thatdata = [3,2,1]
    >>> with HDF5Handler('mydata.hdf5') as handler:
    ...     handler.put(thisdata, "/at/this/location/")
    ...     handler.put(thatdata, "/at/that/location/")
    ...
    >>>


    Another Example
    ---------------

    >>> with HDF5Handler('mydata.hdf5') as handler:
    ...     a_scalar = 1.0
    ...     a_list = [1, 2, 3]
    ...     a_numpy_scalar = numpy.float16(1.0)
    ...     an_ndarray = numpy.arange([1, 2, 3])
    ...     for i in range(5):
    ...         handler.put(a_scalar, '/somepath/scalars')
    ...         handler.put(a_list, '/somepath/lists')
    ...         handler.put(a_numpy_scalar, '/numpies/scalars')
    ...         handler.put(an_ndarray, '/numpies/arrays')
    ...
    >>>

    Since the second argument of handler.put is simply a POSIX-style path,
    this will structure your HDF5 file with the following hierarchy:

        /
        ├── numpies      <-- h5py.Group
        │   ├── arrays      <-- h5py.Dataset
        │   └── scalars     <-- h5py.Dataset
        └── somepath
            ├── lists
            └── scalars

    So Datasets and Groups are quite analogous to Files and Folders.

    #TODO: open mydata.hdf5 and show it indeed contains the data.

    """

    def __init__(self, filename, mode='w', prefix=None):
        """
        Parameters
        ----------
        filename : str
            filename of the hdf5 file.

        mode : str
           Python mode to open file. The mode can be 'w' or 'a' for writing or
           appending. #TODO, check if 'a' mode really works...

        prefix : str
           #TODO explain prefix, and show typical use case.

        """
        self.filename = filename
        self.mode = mode
        self.prefix = prefix

        self.index = dict()
        self.index_converters = dict()

    def __enter__(self):
        self.file = h5py.File(self.filename, self.mode)
        return self

    def __exit__(self, extype, exvalue, traceback):
        self.flushbuffers()
        self.file.close()
        return False

    def put(self, data, dset_path, **kwargs):
        """

        Parameters
        ----------
        data : any valid data type.
            What is meant by "valid" here, is that <data> must be
            convertible with:

            >>> numpy.array(data)

            so this includes things such as:

                scalars :  bool, int, float
                           numpy.int, numpy.float, etc..

                lists   : [int, int, ...]
                          [(float, float), (float, float)]

                tuples  : (float, float, ...)

            However, "valid" in the HDF5Handler also means <data> must also
            be numeric.  This means that the following should not throw a
            TypeError:

            >>> numpy.array(data)/1.0

            Which it will (or should), if <data> contains strings.


        dset_path : str
            unix-style path ( 'group/datasetname' )

        Valid keyword arguments are:

        dtype
        chunksize
        blockfactor
        """

        if self.prefix:
           fulldsetpath = self.prefix+dset_path
        else:
           fulldsetpath = dset_path

        try:
            converter = self.index_converters[fulldsetpath]
            ndarray = converter(data)
            self.index[fulldsetpath].append_to_dbuffer(ndarray)
        except KeyError:
            self.create_dset(data, fulldsetpath, **kwargs)
            self.put(data, dset_path, **kwargs)


    def create_dset(self, data, dset_path, chunksize=1000, blockfactor=100,
                    dtype='float64'):
        """
        Define h5py dataset parameters here.

        Parameters
        ----------
        dset_pathi : str
            A POSIX-style path which will be used as the location for the h5py
            dataset. For example:

        data: any valid data. See HDF5Handler.put.__doc__

        blockfactor : int
            Used to calculate blocksize. (blocksize = blockfactor*chunksize)

        chunksize : int
            Determines the buffersize. (e.g.: if chunksize = 1000, the buffer
            will be written to the dataset after a 1000 HDF5Handler.put()
            calls. You want to make sure that the buffersize is between
            10KiB - 1 MiB = 1048576 bytes.

            This has serious performance implications if chosen too big or
            too small, so I'll repeat that:

            MAKE SURE YOU CHOOSE YOUR CHUNKSIZE SUCH THAT THE BUFFER
            DOES NOT EXCEED 1048576 bytes.

            See h5py docs on chunked storage for more info:
            http://docs.h5py.org/en/latest/high/dataset.html#chunked-storage

            #TODO: Show an example of how you would approximate a good chunksize

        dtype : str
            One of numpy's dtypes.
            int8
            int16
            float16
            float32
            etc.

        """
        arr_shape = get_shape(data)
        converter = get_ndarray_converter(data)

        blocksize = blockfactor * chunksize

        chunkshape = sum(((chunksize,), arr_shape), ())
        maxshape = sum(((None,), arr_shape), ())

        dsetkw = dict(chunks=chunkshape, maxshape=maxshape, dtype=dtype)
        init_shape = sum(((blocksize,), arr_shape), ())
        dset = self.file.create_dataset(dset_path, shape=init_shape, **dsetkw)

        self.index.update({dset_path: Dataset(dset)})
        self.index_converters.update({dset_path: converter})

    def flushbuffers(self):
        """
        When the number of handler.put calls is not a multiple of buffersize,
        then there will be unwritten arrays in dbuffer, since dbuffer is only
        written when it is full. Call this method to write unwritten arrays in
        all of the dbuffers.
        """
        for dset in self.index.values():
            dset.flush()

    #TODO: a method to easily add a comment to the attrs of a dataset.
    def add_comment(self):
        pass

    #TODO: an option to enable one of the lossless compression filters
    # supported by h5py: gzip, lzf, szip
    def compress_with(self):
        pass


class Dataset(object):
    def __init__(self, dset):
        """
        Parameters
        ----------
        dset: h5py Dataset

        """
        self.dset = dset
        self.chunkcounter = 0
        self.blockcounter = 0
        self.chunksize = dset.chunks[0]
        self.blocksize = dset.shape[0]
        self.arr_shape = dset.shape[1:]

        self.dbuffer = list()

    def append_to_dbuffer(self, array):
        """
        Parameters
        ----------
        array: ndarray

        """
        self.dbuffer.append(array)

        if len(self.dbuffer) == self.chunksize: # THEN WRITE AND CLEAR BUFFER
            begin = self.blockcounter*self.blocksize + \
                    self.chunkcounter*self.chunksize
            end = begin + self.chunksize
            dbuffer_ndarray = numpy.array(self.dbuffer)
            self.dset[begin:end, ...] = dbuffer_ndarray # WRITES BUFFER
            self.dbuffer = list()                       # CLEARS BUFFER

            if end == self.dset.shape[0]: #BLOCK IS FULL --> CREATE NEW BLOCK
                new_shape = sum(((end+self.blocksize,), self.arr_shape), ())
                self.dset.resize(new_shape)
                self.blockcounter += 1
                self.chunkcounter = 0
            else:
                self.chunkcounter += 1
        else:
            pass #wait till dbuffer is 'full'

    def flush(self, trim=True):
        dbuffer = self.dbuffer

        dbuffer_ndarray = numpy.array(dbuffer)

        begin = self.blockcounter*self.blocksize +\
                self.chunkcounter*self.chunksize

        end = begin + len(dbuffer)
        self.dset[begin:end, ...] = dbuffer_ndarray
        self.dbuffer = list()

        if trim:
            new_shape = sum(((end,), self.arr_shape), ())
            self.dset.resize(new_shape)


def get_ndarray_converter(data):
    """
    get_ndarray_converter will throw an exception if the data is not "numeric".

    Otherwise, the following applies:

    If the data is of type (numpy.ndarray | int | float, bool), this returns the
    identity function, otherwise it returns numpy.ndarray (as in the function)

    Parameters
    ----------
    data: any valid data format. See HDF5Handler.__doc__

    Return
    ------
    identity OR numpy.array
    """

    try:
        numpy.array(data)/1.0
    except TypeError:
        raise Exception("{data} contains non-numeric objects.".format(data))

    def identity(x):
        return x

    if isinstance(data, numpy.ndarray):
        return identity

    elif isinstance(data, (list, tuple)):
        return numpy.array

    elif isinstance(data, (int, float, bool)):
        return identity

    else:
        raise Exception("{data} could not be converted to ndarray.".format(data))

def get_shape(data):
    """
    Parameters
    ----------
    data: any valid data format. See HDF5Handler.__doc__

    Return
    ------
    returns () if it is a scalar, else it returns numpy.array(data).shape """

    if isinstance(int, float):
        return ()
    else:
        return numpy.array(data).shape
