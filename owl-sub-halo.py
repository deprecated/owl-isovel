import numpy as np
from astropy.io import fits

MARGIN = 20
FILENAME = 'imslit-median.fits'
hdu, = fits.open(FILENAME)
left_profile = hdu.data[:, :MARGIN].mean(axis=1, keepdims=True)
right_profile = hdu.data[:, -MARGIN:].mean(axis=1, keepdims=True)
halo_profile = 0.5*(left_profile + right_profile)
hdu.data -= halo_profile

hdu.writeto(FILENAME.replace('.fits', '-sub.fits'), clobber=True)
