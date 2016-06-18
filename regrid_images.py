import numpy as np
from scipy.interpolate import griddata
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from owl_utils import DATADIR

#
# First set up WCS for the output image
# We use capital letters for the output variables
#

NX, NY = 512, 512
# 0.5 arcsec pixels
dRA, dDec = -0.5/3600., 0.5/3600.
# Center on central star of NGC 3587
RA0, Dec0 = 168.6985, 55.019408
W = WCS(naxis=2)
W.wcs.cdelt = [dRA, dDec]
W.wcs.crpix = [0.5*(1 + NX), 0.5*(1 + NY)]
W.wcs.crval = [RA0, Dec0]
W.wcs.ctype = ['RA---TAN', 'DEC--TAN']

outimage = np.zeros((NY, NX))
# Create world coord arrays for output image
II, JJ = np.meshgrid(np.arange(NX), np.arange(NY))
RA, Dec = W.all_pix2world(II, JJ, 0)

#
# Read in the list of slits
#
table = Table.read('list-of-images.tab', format='ascii.tab')

for row in table:
    hdu, = fits.open(DATADIR +'/' + row['filename'])
    image = (hdu.data - row['bias']) / (row['core'] - row['bias'])
    outfilename = 'imslit-{}.fits'.format(row['index'])
    ny, nx = image.shape
    hdu.header.remove('@EPOCH')
    w = WCS(hdu.header)
    # Create world coord arrays for output image
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    ra, dec = w.all_pix2world(ii, jj, 0)
    # Do the interpolation
    points = np.array(zip(ra.ravel(), dec.ravel()))
    xi = np.array(zip(RA.ravel(), Dec.ravel()))
    outimage = griddata(points, image.ravel(), xi, method='nearest').reshape((NY, NX))
    # Save the output image
    fits.PrimaryHDU(header=W.to_header(), data=outimage).writeto(outfilename, clobber=True)
