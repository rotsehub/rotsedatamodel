'''
Created on Nov 26, 2017

@author: daniel
'''

# STAT is not a recarray, but a 1d ndarray

from astropy.io import fits as pyfits
from scipy.io import readsav
from astropy.io.fits.column import _dtype_to_recformat
import numpy as np
from functools import reduce
import operator
import os


# mul: multiplies the elements of a list.
mul = lambda x: reduce(operator.mul, x, 1)

NUMPY2FITS = pyfits.column.NUMPY2FITS


def bytes2str(b):
    ''' converts bytes to strings to be used in numpy vectorize.

    Args:
        b: a byte.

    Process:
        Decodes the byte into a string using utf-8.

    Returns:
        A string.
    '''
    try:
        return b.decode("utf-8")
    except Exception:
        return b


def getfile(filename):
    ''' Reads a MATCH structured filename into numpy array as dict.
    '''
    fdat = readsav(filename, python_dict=True)
    return fdat


def getmatch(filename):
    ''' fetches the 'match' field within a MATCH structured file.

    Args:
        filename: path to MATCH structured file.

    Process:
        Reads the MATCH structured file.
        Validates that the file read is a dictionary.
        Fetches the 'match' field within the file.

    Returns:
        The 'match' field within the file.
    '''
    fdat = getfile(filename)
    assert isinstance(fdat, dict), "Received non-dict match structure. Make sure the use of python_dict when reading match files"
    rmatch = fdat['match']
    return rmatch


def recarray2bin(rec):
    ''' combines recarray fields into columns within a FITS BinTableHDU.
    fields that are recarray themselves are converted to a separate BinTableHDU.

    Args:
        rec: numpy recarray.

    Process:
        Scans the fields within the recarray.
        Fields that are not recarrays are combined as columns in a BinTableHDU.
        Fields that are recarrays are separated into their own BinTableHDU.

    Returns:
        A list of all bins created from rec.
    '''
    columns = []
    bins = []
    for field in rec.dtype.names:
        data = rec[field]
        result = field2column(data, field)
        if isinstance(result, pyfits.Column):
            columns.append(result)
        else:
            bins.extend(result)
    cbin = cols2hdu(columns)
    allbins = [cbin] + bins
    return allbins


FORCE_FMT = {
    'DRA': 'J',
    'DDEC': 'J',
    'EXPTIME': 'D',
    'CAMTEMP': 'D',
    'MEDIAN': 'D',
    'PKT_T': 'D',
    'TRIG_RA': 'D',
    'TRIG_DEC': 'D',
    'TRIG_ERR': 'D',
    'INTEN': 'D',
    'MOUNTERR': 'D',
    'TEMPOUT': 'D',
    'WINDSPD': 'D',
    'WINDDIR': 'D',
    'HUMIDITY': 'D',
    'RAC': 'D',
    'DECC': 'D',
}


def numpy2fits(obj, name):
    ''' converts a dtype string into its corresponding fits format type.

    Args:
        obj: numpy ndarray.

    Process:
        Uses NUMPY2FITS mapping to fetch the corresponding format.
        If it fails it tries to convert an object to a string,
        or recursively takes the first element of a ndarray.
        When converting bytes, objects are transformed into strings.

    Returns:
        New format type and obj.
    '''
    fmt = obj.dtype
    dtype = str(fmt)
    dtype = _dtype_to_recformat(dtype)[0]

    try:
        ftype = NUMPY2FITS[dtype]
        # ftype = FORCE_FMT.get(name, ftype)
    except Exception as e:
        if dtype.startswith('O'):
            ftype = 'A'
            vfunct = np.vectorize(bytes2str)
            obj = vfunct(obj)
        elif isinstance(obj, np.ndarray):
            index = [0] * len(obj.shape)
            ftype, obj = numpy2fits(obj[tuple(index)], name)
        else:
            raise RuntimeError("Failed NUMPY2FITS dtype: {}".format(dtype)) from e
    return ftype, obj


def A_size(obj):
    ''' Calculates size associated with a string in a nested ndarray, the FITS format A.
    '''
    if isinstance(obj, np.ndarray):
        index = [0] * len(obj.shape)
        return A_size(obj[tuple(index)])
    return len(obj)


def array2column(data, name):
    ''' converts a numpy ndarray that is a field of recarray into a FITS Column.

    Args:
        data: ndarray of simple elements or an unidimensional ndarray of ndarrays of simple elements
        name: name of the recarray field associated with the data.

    Process:
        Identify if the ndarray is nested.
        If it is, convert the top-level unidimensional array into an array structure understood by FITS.
        Then it gets the format of the elements using numpy2fits.
        If the basic element is a string, it factors the length of the string into the format.
        Then it creates a FITS column using the generated format and dimensions.

    Returns:
        Newly generated FITS column.
    '''
    if isinstance(data[0], np.ndarray):
        data = np.array([i for i in data])

    try:
        ftype, data = numpy2fits(data, name)
    except Exception as e:
        raise RuntimeError("Failed to create column: {}; {}".format(name, str(data.dtype))) from e

    if ftype == "A":
        factor = A_size(data)
        index = 0
        if len(data.shape) > 1:
            index = 1
        total = mul(data.shape[index:])*factor
        col_format = '{total}{type}'.format(total=total, type=ftype)
        dims = list(map(str, reversed(data.shape[1:])))
        col_dim = "({}, {})".format(factor, ", ".join(dims))
        '''
        col_format = '{shape}{type}'.format(shape=mul(data.shape[1:])*factor, type=ftype)
        dims = [str(factor)] + list(map(str, reversed(data.shape[1:])))
        col_dim = "({})".format(", ".join(dims))
        '''
    else:
        col_format = '{shape}{type}'.format(shape=mul(data.shape[1:]), type=ftype)
        col_dim = "({})".format(", ".join(map(str, reversed(data.shape[1:]))))

    try:
        column = pyfits.Column(name=name, format=col_format,
                               array=data, dim=col_dim)
    except Exception as e:
        raise RuntimeError("failed to convert {} to column. dtype: {}".format(name, data.dtype)) from e

    # TODO: (Daniel) Show Dr. Kehoe
    ''' spacing in str
    if name in ['CAM_ID', 'CUNIT1', 'CUNIT2', 'OBSTYPE']:
        print(name)
        print(data)
        print(column)
        print(column.array)
    '''

    return column


def field2column(value, name):
    ''' Converts a recarray field into a FITS column or BinTableHDU.

    Args:
        value: field of a recarray.
        name: name of value.

    Process:
        If value contains a recarray nested in a ndarray, it uses recarray2bin to create a BinTableHDU.
        If value is simply nested ndarrays, it uses array2column to create a FITS column.

    Returns:
        Newly generated FITS column or BinTableHDU.
    '''
    if isinstance(value[0], np.recarray):
        result = recarray2bin(value[0])
    else:
        result = array2column(data=value, name=name)
    return result


def cols2hdu(columns):
    ''' Converts FITS columns into a BinTableHDU.
    '''
    for column in columns:
        try:
            pyfits.BinTableHDU.from_columns(columns=[column])
        except Exception:
            print('Failed at {}'.format(column.name))
    fits = pyfits.BinTableHDU.from_columns(columns=columns)
    return fits


def bins2hdulist(match):
    ''' creates a HDUList of all BinTableHDUs.
    
    Args:
        match: field of recarray

    Process:
        Creates a PrimaryHDU header with a comment.
        Creates the BinTableHDU from match.
        Combines the Primary HDU with the BinTableHDUs into a list.
        Creates a HDUList of all the tables.

    Returns:
        Newly generated HDUList of tables from match.
    '''
    prihdr = pyfits.Header()
    prihdu = pyfits.PrimaryHDU(header=prihdr)
    prihdr['COMMENT'] = "Automatic convertion of MATCH to FITS."

    bins = recarray2bin(match)
    bins = [prihdu] + bins

    hdulist = pyfits.HDUList(bins)
    return hdulist


def match2fits(datfile, fitspath=None):
    ''' converts a file with MATCH structure into a FITS structured file.

    Args:
        datfile: path to MATCH structured file.
        fitspath: optional path or target directory of file to be created. Defaults to datfile.fit.

    Process:
        Compute the target FITS file's default name.
        Compose the target FITS file from the name of the MATCH file.
        Loads the MATCH structured file into numpy structures.
        Creates FITS BinTableHDUs from recarrays and columns for ndarrays.
        Saves the BinTableHDUs into a FITS file.

    Returns:
        Path to the FITS file.
    '''
    def fitsname2match(match_file):
        ''' Retrieves default target fitspath from datfile.
        '''
        fitspath = match_file
        if match_file.endswith('dat'):
            fitspath = match_file[:-3]
        elif datfile.endswith('datc'):
            fitspath = match_file[:-4]
        fitspath += 'fit'
        return fitspath

    if fitspath is None:
        fitspath = fitsname2match(datfile)
    elif os.path.isdir(fitspath):
        filename = os.path.basename(datfile)
        fitsname = fitsname2match(filename)
        fitspath = os.path.join(fitspath, fitsname)

    m = getmatch(datfile)
    thdulist = bins2hdulist(m)

    # thdulist[1].name = 'MATCH'
    thdulist.writeto(fitspath, overwrite=True)
    return fitspath


def multimatch2fits(*datfile, fitspath=None):
    ''' converts multiple MATCH structured files into FITS structured files.

    Args:
        datfile: list of paths to MATCH structured files.
        fitspath: existing target directory in which the FITS files will be created.
            If there is only 1 datfile, fitspath would be considered a target file if it is not a directory.

    Process:
        Validates that if datfile is a list of multiple files and a fits path is provided, fitspath is a directory.
        Converts each MATCH file in datfile into a FITS file.
        Creates a list of the paths of the FITS files created.
    Returns:
        List of the paths of the FITS files created.
    '''
    result = []
    if fitspath is not None:
        if not os.path.isdir(fitspath) and len(datfile) > 1:
            raise RuntimeError("fitspath must be an existing directory when converting multiple MATCH files")

    for match in datfile:
        fits = match2fits(match, fitspath)
        result.append(fits)

    return result


def compare(hdu, match, value):
    vfits = hdu.data.field(value)
    print(vfits.shape)
    rmatch = match  # ["match"]
    v = rmatch[value][0]
    print('Retrieved the same:', np.array_equal(v, vfits))
    print(vfits[0, 1], vfits[1, 0])
    print(v[0, 1], v[1, 0])


if __name__ == "__main__":
    from scipy.io import readsav

    heredir = os.path.dirname(os.path.abspath(__file__))
    basedir = os.path.dirname(heredir)
    projdir = os.path.dirname(basedir)
    datdir = os.path.join(projdir, 'dat')

    filename = '000409_xtetrans_1a_match.dat'
    # filename = '000929_sky0001_1a_match.datc'
    datfile = os.path.join(datdir, filename)

    # fitsfile = "test.fit"
    # match2fits(datfile=datfile, filepath=fitsfile)
    match2fits(datfile=datfile)
