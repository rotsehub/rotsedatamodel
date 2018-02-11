AUTHOR: Daniel Sela

This is the description of a conversion tool from ROTSE's MATCH format to FITS formatting.
=============================================================================
COMMENTS/SUGGESTIONS: danielsela42@gmail.com

ACKNOWLEDGEMENT: Arnon Sela, Dr. Robert Kehoe, Govinda Dhungana

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Objective:
To create a utility program that would be used to convert MATCH format into the more used FITS format.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Programs:
=========

match2fits
-----------

creates a PDF file instead of a postscript of all the stars within the designated file. The file types supported are .datc, .dat, and .fit.

Parameters:
    --match (-m): paths to MATCH structured files.
    --fits (-f): existing target directory in which the FITS files will be created. Or a target file in the case of a single given MATCH file.

To run:

    match2fits -m FILE(S) -f FILE(DIR)

Example:

    match2fits -match 000409_xtetrans_1a_match.dat 000409_xtetrans_1b_match.dat -fits example.fit

For more information run match2fits -h (or --help)
