# [[file:teresa-owl.org::*%20Program%20owl-slit-calibrate.py][\[2/3\]\ Program\ owl-slit-calibrate\.py:1]]
open -n -a SAOImage\ DS9 --args -title spectra
# \[2/3\]\ Program\ owl-slit-calibrate\.py:1 ends here

# [[file:teresa-owl.org::*%20Program%20owl-slit-calibrate.py][\[2/3\]\ Program\ owl-slit-calibrate\.py:2]]
import os
import sys
import numpy as np
import astropy
from astropy.table import Table, hstack
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from matplotlib import pyplot as plt
import seaborn as sns
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.modeling import models, fitting
from owl_utils import (DATADIR, slit_profile, extract_profile, 
                       find_slit_coords, make_slit_wcs, remove_bg_and_regularize)

restwavs = {'ha': 6562.79, 'nii': 6583.45}

# Position of star
RA0, Dec0 = 168.69895, 55.019147

def fit_cheb(x, y, npoly=3, mask=None):
    """Fits a Chebyshev poly to y(x) and returns fitted y-values"""
    fitter = fitting.LinearLSQFitter()
    p_init = models.Chebyshev1D(npoly, domain=[x.min(), x.max()])
    if mask is None:
        mask = np.ones_like(x).astype(bool)
    p = fitter(p_init, x[mask], y[mask])
    print(p)
    return p(x)

sns.set_palette('RdPu_d', 3)
def make_three_plots(spec, calib, prefix, niirat=None):
    assert spec.shape == calib.shape
    fig, axes = plt.subplots(3, 1)


    ypix = np.arange(len(calib))
    m = np.isfinite(spec) & np.isfinite(calib)
    spec = spec[m]
    calib = calib[m]
    ypix = ypix[m]
    if niirat is not None:
        niirat = niirat[m]

    vmin, vmax = -0.1, np.percentile(calib, 95) + 2*calib.std()
    ratio = spec/calib
    mask = (spec > np.percentile(spec, 25)) & (ratio > 0.5) & (ratio < 2.0)
    # mask = (ypix > 10.0) & (ypix < ypix.max() - 10.0) \
    #        & (ratio > np.median(ratio) - 2*ratio.std()) \
    #        & (ratio < np.median(ratio) + 2*ratio.std()) 
    try:
        ratio_fit = fit_cheb(ypix, ratio, mask=mask, npoly=1)
    except:
        ratio_fit = np.ones_like(ypix)

    alpha = 0.8

    # First, plot two profiles against each other to check for zero-point offsets
    axes[0].plot(calib, spec/ratio_fit, '.', alpha=alpha)
    axes[0].plot([vmin, vmax], [vmin, vmax], '-', alpha=alpha)
    axes[0].set_xlim(vmin, vmax)
    axes[0].set_ylim(vmin, vmax)
    axes[0].set_xlabel('Calibration Image')
    axes[0].set_ylabel('Corrected Integrated Spectrum')

    # Second, plot each against slit pixel to check spatial offset
    axes[1].plot(ypix, calib, alpha=alpha, label='Calibration Image')
    axes[1].plot(ypix, spec/ratio_fit, alpha=alpha, lw=1.0,
                 label='Corrected Integrated Spectrum')
    axes[1].plot(ypix, spec, alpha=alpha, lw=0.5,
                 label='Uncorrected Integrated Spectrum')
    axes[1].set_xlim(0.0, ypix.max())
    axes[1].set_ylim(vmin, vmax)
    axes[1].legend(fontsize='xx-small', loc='upper right')
    axes[1].set_xlabel('Slit pixel')
    axes[1].set_ylabel('Profile')

    # Third, plot ratio to look for spatial trends
    axes[2].plot(ypix, ratio, alpha=alpha)
    axes[2].plot(ypix, ratio_fit, alpha=alpha)
    if niirat is not None:
        axes[2].plot(ypix, niirat, 'b', lw=0.5, alpha=0.5)
    axes[2].set_xlim(0.0, ypix.max())
    axes[2].set_ylim(0.0, 2.5)
    axes[2].set_xlabel('Slit pixel')
    axes[2].set_ylabel('Ratio: Spec / Calib')


    fig.set_size_inches(5, 8)
    fig.tight_layout()
    fig.savefig(prefix+'.png', dpi=300)

    return ratio_fit


hatab = Table.read('spectra-ha.tab', format='ascii.tab')
niitab = Table.read('spectra-nii.tab', format='ascii.tab')
imtab = Table.read('list-of-images.tab', format='ascii.tab')
slittab = Table.read('slit-positions.tab', format='ascii.tab')
table = hstack([hatab, niitab, imtab, slittab],
               table_names=['ha', 'nii', 'im', 'slit'])
# Photometric reference image
photom, = fits.open('imslit-median-sub.fits')
wphot = WCS(photom.header)

nii_ha_dict = {'radius': [], 'ratio': [], 'weight': []}
for row in table:
    ha_hdu, = fits.open(DATADIR +'/SPMha/' + row['file_ha'])
    nii_hdu, = fits.open(DATADIR +'/SPMnii/' + row['file_nii'])
    im_hdu, = fits.open(DATADIR +'/' + row['filename'])
    im_hdu.header.remove('@EPOCH')



    slit_coords = find_slit_coords(row, im_hdu.header, ha_hdu.header)
    calib_profile = slit_profile(slit_coords['RA'], slit_coords['Dec'],
                                 photom.data, wphot)

    ha_profile, ha_bg = extract_profile(ha_hdu.data, WCS(ha_hdu.header),
                                        restwavs['ha'], row)
    nii_profile, nii_bg = extract_profile(nii_hdu.data, WCS(nii_hdu.header),
                                          restwavs['nii'], row)


    # Make a fake Ha+[N II] line (really should add continuum too)
    spec_profile = (ha_profile+1.333*nii_profile) # + 3.4*(ha_bg + nii_bg)
    # Zeroth order approximation to calibration
    rat0 = np.nansum(spec_profile)/np.nansum(calib_profile)
    print('Coarse calibration: ratio =', rat0)
    spec_profile /= rat0

    plt_prefix = '{:03d}-calib'.format(row.index+1)
    ratio = make_three_plots(spec_profile, calib_profile,
                             plt_prefix, niirat=nii_profile/ha_profile)

    nii_ha_dict['radius'].extend(
        np.hypot(slit_coords['RA'] - RA0, slit_coords['Dec'] - Dec0)*3600
    )
    nii_ha_dict['ratio'].extend(nii_profile/ha_profile)
    nii_ha_dict['weight'].extend(spec_profile)


    # Save calibrated spectra to files
    for lineid, hdu in [['ha', ha_hdu], ['nii', nii_hdu]]:
        restwav = restwavs[lineid]
        print('Saving', lineid, 'calibrated spectrum')
        # Apply basic calibration zero-point and scale
        hdu.data, _ = remove_bg_and_regularize(hdu.data, WCS(hdu.header),
                                               restwav, row)/rat0
        # Regularize spectral data so that wavelength is x and pos is y
        # This is now done by the bg subtraction function

        # Apply polynomial correction along slit
        if ratio is not None:
            hdu.data /= ratio.mean()
        # Extend in the third dimension (degenerate axis perp to slit)
        hdu.data = hdu.data[None, :, :]

        # Create the WCS object for the calibrated slit spectra
        wslit = make_slit_wcs(row, slit_coords, hdu)
        # Set the rest wavelength for this line
        wslit.wcs.restwav = (restwav*u.Angstrom).to(u.m).value
        # # Remove WCS keywords that might cause problems
        # for i in 1, 2:
        #     for j in 1, 2:
        #         kwd = 'CD{}_{}'.format(i, j)
        #         if kwd in hdu.header:
        #             hdu.header.remove(kwd) 
        # Then update the header with the new WCS structure as the 'A'
        # alternate transform
        hdu.header.update(wslit.to_header(key='A'))
        # Also save the normalization factor as a per-slit weight to use later
        hdu.header['WEIGHT'] = rat0

        # And better not to change the original WCS at all
        # Unless we have transposed the array, which we have to compensate for
        if row['saxis'] == 1:
            for k in ['CRPIX{}', 'CRVAL{}', 'CDELT{}', 'CD{0}_{0}']:
                hdu.header[k.format('1')], hdu.header[k.format('2')] = hdu.header[k.format('2')], hdu.header[k.format('1')] 
        # # And write a bowdlerized version that DS9 can understand as the main WCS
        # hdu.header.update(fixup4ds9(wslit).to_header(key=' '))
        calibfile = 'Calibrated/{}-{}.fits'.format(row['pos_nii'], lineid)
        hdu.writeto(calibfile, clobber=True)


# Now do something with that nii/ha versus radius data
Table(nii_ha_dict).write('owl-nii-ha-ratio.tab', format='ascii.tab')
# \[2/3\]\ Program\ owl-slit-calibrate\.py:2 ends here
