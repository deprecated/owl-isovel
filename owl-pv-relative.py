import glob
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
import astropy.units as u
sys.path.append('/Users/will/Dropbox/OrionWest')
from helio_utils import helio_topo_from_header, waves2vels


speclist = glob.glob('Calibrated/*-*.fits')
# Center on central star of NGC 3587
RA0, Dec0 = 168.6985, 55.019408

for fn in speclist:
    print('Processing', fn)
    spechdu, = fits.open(fn)
    wspec = WCS(spechdu.header, key='A')

    # Eliminate degenerate 3rd dimension from data array and trim off bad bits
    spec2d = spechdu.data[0]

    # Rest wavelengths in m
    if '-nii' in fn:
        wav0 = 6583.45*1e-10
    elif '-ha' in fn:
        wav0 = 6562.79*1e-10
    else:
        raise NotImplementedError('Only Ha and [N II] so far')

    # Convert to heliocentric velocity
    [[wav1, _, _], [wav2, _, _]] = wspec.all_pix2world([[0, 0, 0], [1, 0, 0]], 0)
    [v1, v2] = waves2vels(np.array([wav1, wav2]), wav0, spechdu.header)

    # Find RA and declination offset
    [[_, i0, j0]] = wspec.all_world2pix([[6563.0, RA0, Dec0]], 0)
    [[_, RA1, Dec1]] = wspec.all_pix2world([[0, i0, 0]], 0)

    wnew = WCS(naxis=2)
    wnew.wcs.ctype = ['VHEL', 'LINEAR']
    wnew.wcs.crpix = [1, i0+1]
    wnew.wcs.cunit = ['km/s', 'arcsec']
    wnew.wcs.crval = [v1.to('km/s').value, 3600*(RA1 - RA0)]
    wnew.wcs.cdelt = [(v2 - v1).to('km/s').value,
                      3600*wspec.wcs.cdelt[1]*wspec.wcs.pc[1,1]]

    decoffset =  3600*(Dec1 - Dec0)
    decstring = '-y{:+04d}'.format(int(decoffset))
    newhdr = wnew.to_header()
    new_fn = fn.replace('Calibrated/', 'Offset/')
    new_fn = new_fn.replace('.fits', decstring + '.fits')
    fits.PrimaryHDU(data=spec2d, header=newhdr).writeto(new_fn, clobber=True)
