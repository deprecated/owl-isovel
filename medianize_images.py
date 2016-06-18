import numpy as np
from astropy.io import fits

NIM = 12
imlist = []
fnlist = []
for i in range(1, NIM+1):
    fitsname = 'imslit-pos{:02d}.fits'.format(i)
    hdu, = fits.open(fitsname)
    imlist.append(hdu.data)
    fnlist.append(fitsname)
imstack = np.dstack(imlist)
median = np.median(imstack, axis=-1)
fits.PrimaryHDU(header=hdu.header,
                data=median).writeto('imslit-median.fits', clobber=True)

for im, fn in zip(imlist, fnlist):
    outname = fn.replace('imslit', 'inslit-ratio')
    fits.PrimaryHDU(header=hdu.header,
                    data=im/median).writeto(outname, clobber=True)
