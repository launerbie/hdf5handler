#!/usr/bin/env python

"""
Numpy supports a much greater variety of numerical types than Python does.
This section shows which are available, and how to modify an array’s
data-type.mroData type Description

bool_        Boolean (True or False) stored as a byte
int_         Default integer type (same as C long; normally either int64 or int32)
intc         Identical to C int (normally int32 or int64)
intp         Integer used for indexing (same as C ssize_t; normally either int32 or int64)
int8         Byte (-128 to 127)
int16        Integer (-32768 to 32767)
int32        Integer (-2147483648 to 2147483647)
int64        Integer (-9223372036854775808 to 9223372036854775807)
uint8        Unsigned integer (0 to 255)
uint1        Unsigned integer (0 to 65535)
uint32       Unsigned integer (0 to 4294967295)
uint64       Unsigned integer (0 to 18446744073709551615)
float_       Shorthand for float64.
float16      Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
float32      Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
float64      Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
complex_     Shorthand for complex128.
complex64    Complex number, represented by two 32-bit floats (real and imaginary components)
complex128   Complex number, represented by two 64-bit floats (real and imaginary components)
"""

#Note: not all of the above types have been tested.

import h5py
from hdf5handler import HDF5Handler

FILENAME = 'alotofdata.h5'

with HDF5Handler(FILENAME, 'w') as h:
    for i in range(200):
        h.put(i, 'default')

    for i in range(200):
        h.put(i, 'myinteights', dtype='int8')

    for i in range(200):
        h.put(i, 'myintsixteens', dtype='int16')

    for i in range(200):
        h.put(i, 'myintthirtytwos', dtype='int32')

    for i in range(200):
        h.put(i, 'myintsixtyfours', dtype='int64')

    for i in range(200):
        h.put(i, 'myfloatsixteens', dtype='float16')

    for i in range(200):
        h.put(i, 'myfloat64', dtype='float64')

# Exiting the context manager will close the hdf5 file, thus we can now
# open and read the data.

f = h5py.File(FILENAME)

print(list(f.keys()))

for key in f:
    n = f[key].value
    print(key, n.dtype, n.shape, n.nbytes, )




