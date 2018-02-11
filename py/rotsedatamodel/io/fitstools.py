'''
Created on Dec 20, 2017

@author: daniel
'''

from .nptools import add_recarray_field
from astropy.io import fits as pyfits


def readfits(filepath):
    ''' creates a numpy based structure from a FITS file generated by match2fits.

    Args:
        filepath: path to FITS file.

    Process:
        Opens FITS file and reads each of its BinTableHDUs.
        Uses add_recarray_field to create a combined recarray using the BinTableHDUs.

    Returns:
        A numpy recarray.
    '''
    with pyfits.open(filepath, memmap=True) as hdus:
        match = hdus[1].data
        stat = hdus[2].data
        map_ = hdus[3].data
    stat_map = [('STAT', stat), ('MAP', map_)]
    new_match = add_recarray_field(match, stat_map)
    return new_match


if __name__ == '__main__':
    import os

    heredir = os.path.dirname(os.path.abspath(__file__))
    basedir = os.path.dirname(heredir)
    projdir = os.path.dirname(basedir)
    datdir = os.path.join(projdir, 'dat')

    filename = '000409_xtetrans_1a_match.fit'
    fitsfile = os.path.join(datdir, filename)

    fits = readfits(fitsfile)
    print(fits.dtype)
    '''
    from concepts.mappingfits import elements

    elements(fitsfile)
    print(pyfits.info(fitsfile))
    '''