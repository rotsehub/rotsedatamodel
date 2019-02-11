'''
Created on Dec 19, 2017

@author: Daniel
'''

import numpy as np


def case_insensative_recarray(dtype):
    ''' for single named fields, create a tuple of the lowercase and uppercase versions.

    Args:
        dtype: np data type

    Process:
        For each dtype element, checks if it is single-named.
        If it is, replace the name with a tuple of lower and upper cases.

    Returns:
        New dtype with a new naming convention.
    '''
    ndtype = []
    for i in dtype.descr:
        name = i[0]
        if isinstance(name, str):
            i = tuple([(name.lower(), name.upper())] + list(i[1:]))
        ndtype += [i]
    return np.dtype(ndtype)


def add_recarray_field(recarray, narr):
    ''' append new field in a recarray

    Args:
        recarray : numpy recarray
        narr: A list of tuples (name, array)
            name : (str) name of field to be appended
            array : numpy array or recarray to be appended

    Process:
        Creates a list of dtypes corresponding to fields narr.
        Creates an empty recarray with old and new fields.
        Copies fields of source recarray into the new recarray.
        Copies new fields into the new recarray.

    Returns:
        numpy recarray appended
    '''
    # creates dtype of newfields
    newfields = []
    for name, array in narr:
        arr = np.asarray(array)
        numpy_dtype = arr.dtype
        if isinstance(array, np.recarray):
            numpy_dtype = case_insensative_recarray(numpy_dtype)
        newfields += [((name.lower(), name.upper()), numpy_dtype, arr.shape)]
    # make sure base recarray and both lower and upper names
    base_dtype = case_insensative_recarray(recarray.dtype)

    # create new empty recarray based on source, with added new field
    newdtype = np.dtype(base_dtype.descr + newfields)
    newrec = np.empty(recarray.shape, dtype=newdtype)

    # copy source fields to new recarray
    for field in recarray.dtype.fields:
        newrec[field] = recarray[field]

    # copy new field to its place in new recarray
    for name, array in narr:
        try:
            arr = np.asarray(array)
            newrec[name] = arr
        except Exception as e:
            raise RuntimeError("Failed to append {}; shape: {}; dtype: {}.".format(name, arr.shape, newdtype)) from e
    return newrec


def create_struct(arrays):
    ''' combines arrays into recarray.
    Args:
        arrys: list of (name, ndarray) tuples.
    '''

    # create new dtype for the recarray
    fields = []
    for name, array in arrays:
        fields += [(name, array.dtype, array.shape)]

    head_name, head_array = arrays[0]
    dtype = np.dtype(fields)
    rec = np.empty(head_array.shape, dtype=dtype)

    # copy source fields to new recarray
    for field, array in arrays:
        rec[field] = array

    return rec


if __name__ == '__main__':
    x = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', float), ('y', int)])
    y = [('a', (20, 10)), ('b', (30, 50))]
    z = add_recarray_field(x, y)
    print(z.dtype, z)

    shape = (3, 4, 5)
    a = np.zeros(shape, dtype=np.float32)
    b = np.zeros(shape, dtype=np.float32)
    ab_list = [('a', a), ('b', b)]
    ab = create_struct(ab_list)
    print(ab)
    c = np.zeros(shape, dtype=np.int32)
    abc_list = ab_list + [('c', c)]
    abc = create_struct(abc_list)

    d = 100
    new = np.empty(shape, dtype=[('a', 'f4', shape), ('b', 'f4', shape), ('c', 'i4', shape), ('d', 'i4', 1)])
    print(new)
    print(new.dtype)
