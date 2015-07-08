#!/usr/bin/env python

from hdf5handler import HDF5Handler

handler = HDF5Handler('mydata.hdf5')
handler.open()

for i in range(100):
    handler.put(i, 'numbers')

handler.close()


