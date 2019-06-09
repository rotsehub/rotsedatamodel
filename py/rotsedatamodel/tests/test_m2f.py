'''
Created on Dec 13, 2017

@author: arnon
'''


import unittest as ut
import numpy as np
from rotsedatamodel.match2fits import multimatch2fits, getmatch
from rotsedatamodel.io.fitstools import readfits


def remove_spaces(s):
    # remove spaces in a string
    return s.strip()


def compare_recarray(rec1, rec2):
    ''' compares 2 recarrays by looking at their individual fields

    Args:
        rec1, rec2: 2 recarrays to compare

    Process:
        Filters fields by using uppercase characters in the name.
        Examining the shape to identify if one level deeper needs to be compared in rec1 or rec2.
        Special text fields are stripped from spaces, since once built into a FITS column, spaces are removed.
        Adds each field that has different contents to a list.

    Returns:
        A list of the fields with different contents between rec1 and rec2.

    '''
    different = []
    filtered_fields = filter(lambda x: x.upper() == x, rec1.dtype.fields.keys())
    for field in sorted(filtered_fields):
        recfield1 = rec1[field]
        recfield2 = rec2[field]
        if isinstance(recfield1[0], np.recarray):
            diff = compare_recarray(recfield1[0], recfield2[0])
            different += diff
        else:
            specialfields = ['CAM_ID', 'CUNIT1', 'CUNIT2', 'OBSTYPE']
            recindex1 = recfield1
            recindex2 = recfield2
            if isinstance(recfield1[0], np.ndarray):
                recindex1 = recfield1[0]
            vfunct = np.vectorize(remove_spaces)
            if field in specialfields:
                recindex1 = vfunct(recindex1)
            if recindex1.shape != recindex2.shape:
                if not np.array_equal(recindex1, recfield2[0]):
                    different += [field]
            else:
                if not np.array_equal(recindex1, recfield2):
                    different += [field]
    return different


class TestM2F(ut.TestCase):
    def columns(self):
        self.assertTrue('FOO'.isupper())

    def test_match2fits_dat(self):
        ''' run match2fits on a match file.
            read the match file into memory.
            read fits file into memory.
            compare the two memory structures.
        '''
        match_file = '../dat/000409_xtetrans_1a_match.dat'
        # These 3 lines: 73-75 as funct.
        fits_file = multimatch2fits(match_file)
        match = getmatch(match_file)
        fits = readfits(fits_file[0])
        diff = compare_recarray(match, fits)
        self.assertTrue(len(diff) == 0, 'Failed in field: {}'.format(diff))


if __name__ == '__main__':
    ut.main()
