import sys
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel

try:
    suffix = sys.argv[1]
except IndexError:
    print('Usage: {} SUFFIX'.format(sys.argv[0]))

slit_hdus = fits.open('owl-slits-{}.fits'.format(suffix))
slit_wt = slit_hdus['weight'].data
slit_im_wt = slit_hdus['slits'].data
slit_im = slit_hdus['scaled'].data

imhdu, = fits.open('imslit-median-sub.fits')

# Make BG mask by smoothing and thresholding
gauss_kernel = Gaussian2DKernel(5)
im_smooth = convolve_fft(imhdu.data, gauss_kernel)
bgmask = im_smooth < 0.05
median_bg = np.median(imhdu.data[bgmask])
# Arbitrary weight for the new bg pixels, similar to slit weights
bgwt = 3000

# Find slit mask
slit_mask = slit_wt > 0.0
median_slit_bg = np.median(slit_im[slit_mask & bgmask])

# Add scaled BG to multislit image, but only in between slits
fill_mask = (~slit_mask) & bgmask
slit_im_wt[fill_mask] = bgwt*(median_slit_bg/median_bg)*imhdu.data[fill_mask]
slit_wt[fill_mask] = bgwt

slit_hdus['slits'].data = slit_im_wt
slit_hdus['weight'].data = slit_wt
slit_hdus['scaled'].data = slit_im_wt/slit_wt
slit_hdus.writeto('owl-dslits-{}.fits'.format(suffix), clobber=True)
