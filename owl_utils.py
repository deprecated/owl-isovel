import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord
from astropy import units as u

DATADIR = '/Users/will/Dropbox/Papers/LL-Objects/NGC3587'

def slit_profile(ra, dec, image, wcs):
    """
    Find the image intensity for a list of positions (ra and dec)
    """
    xi, yj = wcs.all_world2pix(ra, dec, 0)
    # Find nearest integer pixel
    ii, jj = np.floor(xi + 0.5), np.floor(yj + 0.5)
    print(ra[::100], dec[::100])
    print(ii[::100], jj[::100])
    ny, nx = image.shape
    return np.array([image[j, i] if (i < nx and j < ny) else np.nan for i, j in zip(ii, jj)])

def find_slit_coords(db, hdr, shdr):
    """Find the coordinates of all the pixels along a spectrograph slit

    Input arguments are a dict-like 'db' of hand-measured values (must
    contain 'saxis', 'islit' and 'shift') and a FITS headers 'hdr' from
    the image+slit exposure and 'shdr' from a spectrum exposure

    Returns a dict of 'ds' (slit pixel scale), 'PA' (slit position
    angle), 'RA' (array of RA values in degrees along slit), 'Dec'
    (array of Dec values in degrees along slit)

    """
    jstring = str(db['saxis'])  # which image axis lies along slit
    dRA_arcsec = hdr['CD1_'+jstring]*3600*np.cos(np.radians(hdr['CRVAL2']))
    dDEC_arcsec = hdr['CD2_'+jstring]*3600
    ds = np.hypot(dRA_arcsec, dDEC_arcsec)
    PA = np.degrees(np.arctan2(dRA_arcsec, dDEC_arcsec))

    # Pixel coords of each slit pixel on image (in 0-based convention)
    if jstring == '1':
        # Slit is horizontal in IMAGE coords
        ns = shdr['NAXIS1']
        iarr = np.arange(ns) - float(db['shift'])
        jarr = np.ones(ns)*float(db['islit'])
        try:
            image_binning = hdr['CBIN']
            spec_binning = shdr['CBIN']
        except KeyError:
            image_binning = hdr['CCDXBIN']
            spec_binning = shdr['CCDXBIN']

        # correct for difference in binning between the image+slit and the spectrum
        iarr *= spec_binning/image_binning
    elif jstring == '2':
        # Slit is vertical in IMAGE coords
        ns = shdr['NAXIS2']
        iarr = np.ones(ns)*float(db['islit'])
        jarr = np.arange(ns) - float(db['shift'])
        try:
            image_binning = hdr['RBIN']
            spec_binning = shdr['RBIN']
        except KeyError:
            image_binning = hdr['CCDYBIN']
            spec_binning = shdr['CCDYBIN']

        jarr *= spec_binning/image_binning
    else:
        raise ValueError('Slit axis (saxis) must be 1 or 2')

    print('iarr =', iarr[::100], 'jarr =', jarr[::100])
    # Also correct the nominal slit plate scale
    ds *= spec_binning/image_binning

    # Convert to world coords, using the native frame
    w = WCS(hdr)
    observed_frame = w.wcs.radesys.lower()
    # Note it is vital to ensure the pix2world transformation returns
    # values in the order (RA, Dec), even if the image+slit may have
    # Dec first
    coords = SkyCoord(*w.all_pix2world(iarr, jarr, 0, ra_dec_order=True),
                      unit=(u.deg, u.deg), frame=observed_frame)
    print('coords =', coords[::100])
    print('Binning along slit: image =', image_binning, 'spectrum =', spec_binning)
    # Make sure to return the coords in the ICRS frame
    return {'ds': ds, 'PA': PA,
            'RA': coords.icrs.ra.value,
            'Dec': coords.icrs.dec.value}

def make_slit_wcs(db, slit_coords, spechdu):
    # Input WCS from original spectrum
    wspec = WCS(spechdu.header)
    wspec.fix()

    #
    # First find wavelength scale from the spectrum  
    #

    # For original spectrum, the wavelength and slit axes are 0-based,
    # but in FITS axis order instead of python access order, since
    # that is the way that that the WCS object likes to do it
    ospec_wavaxis = 2 - db['saxis']
    ospec_slitaxis = db['saxis'] - 1

    # The rules are that CDi_j is used if it is present, and only if
    # it is absent should CDELTi be used
    if wspec.wcs.has_cd():
        dwav = wspec.wcs.cd[ospec_wavaxis, ospec_wavaxis]
        # Check that the off-diagonal terms are zero
        assert(wspec.wcs.cd[0, 1] == wspec.wcs.cd[1, 0] == 0.0)
    else:
        dwav = wspec.wcs.cdelt[ospec_wavaxis]
        if wspec.wcs.has_pc():
            # If PCi_j is also present, make sure it is identity matrix
            assert(wspec.wcs.pc == np.eye(2))
    wav0 = wspec.wcs.crval[ospec_wavaxis]
    wavpix0 = wspec.wcs.crpix[ospec_wavaxis]

    #
    # Second, find the displacement scale and ref point from the slit_coords
    #
    # The slit_coords should already be in ICRS frame
    c = SkyCoord(slit_coords['RA'], slit_coords['Dec'], unit=u.deg)
    # Find vector of separations between adjacent pixels
    seps = c[:-1].separation(c[1:])
    # Ditto for the position angles
    PAs = c[:-1].position_angle(c[1:])
    # Check that they are all the same as the first one
    assert(np.allclose(seps/seps[0], 1.0))
    # assert(np.allclose(PAs/PAs[0], 1.0, rtol=1.e-4))
    # Then use the first one as the slit pixel size and PA
    ds, PA, PA_deg = seps[0].deg, PAs.mean().rad, PAs.mean().deg
    # And for the reference values too
    RA0, Dec0 = c[0].ra.deg, c[0].dec.deg

    #
    # Now make a new shiny output WCS, constructed from scratch
    #
    w = WCS(naxis=3)

    # Make use of all the values that we calculated above
    w.wcs.crpix = [wavpix0, 1, 1]
    w.wcs.cdelt = [dwav, ds, ds]
    w.wcs.crval = [wav0, RA0, Dec0]
    # PC order is i_j = [[1_1, 1_2, 1_3], [2_1, 2_2, 2_3], [3_1, 3_2, 3_3]]
    w.wcs.pc = [[1.0, 0.0, 0.0],
                [0.0, np.sin(PA), -np.cos(PA)],
                [0.0, np.cos(PA), np.sin(PA)]]

    #
    # Finally add in auxillary info
    #
    w.wcs.radesys = 'ICRS'
    w.wcs.ctype = ['AWAV', 'RA---TAN', 'DEC--TAN']
    w.wcs.specsys = 'TOPOCENT'
    w.wcs.cunit = [u.Angstrom, u.deg, u.deg]
    w.wcs.name = 'TopoWav'
    w.wcs.cname = ['Observed air wavelength', 'Right Ascension', 'Declination']
    w.wcs.mjdobs = wspec.wcs.mjdobs
    w.wcs.datfix()              # Sets DATE-OBS from MJD-OBS

    # Check the new pixel values
    npix = len(slit_coords['RA'])
    check_coords = pixel_to_skycoord(np.arange(npix), [0]*npix, w, 0)
    # These should be the same as the ICRS coords in slit_coords
    print('New coords:', check_coords[::100])
    print('Displacements in arcsec:', check_coords.separation(c).arcsec[::100])
    # 15 Sep 2015: They seem to be equal to within about 1e-2 arcsec

    return w

def extract_profile(data, wcs, wavrest, db, dw=7.0):
    data, bgdata = remove_bg_and_regularize(data, wcs, wavrest, db)
    # pixel limits for line extraction
    lineslice = wavs2slice([wavrest-dw/2, wavrest+dw/2], wcs, db)
    return data[:, lineslice].sum(axis=1), bgdata.sum(axis=1)


def wavs2slice(wavs, wcs, db):
    """Convert a wavelength interval `wavs` (length-2 sequence) to a slice of the relevant axis`"""
    assert len(wavs) == 2
    isT = db['saxis'] == 1
    if isT:
        _, xpixels = wcs.all_world2pix([0, 0], wavs, 0)
    else:
        xpixels, _ = wcs.all_world2pix(wavs, [0, 0], 0)
    print('Wav:', wavs, 'Pixel:', xpixels)
    i1, i2 = np.maximum(0, (xpixels+0.5).astype(int))
    return slice(min(i1, i2), max(i1, i2))


def remove_bg_and_regularize(data, wcs, wavrest, db, dwbg_in=7.0, dwbg_out=10.0):
    '''
    Transpose data if necessary, and then subtract off the background (blue and red of line)
    '''
    isT = db['saxis'] == 1
    # Make sure array axis order is (position, wavelength)
    if isT:
        data = data.T
    # pixel limits for blue, red bg extraction
    bslice = wavs2slice([wavrest-dwbg_out/2, wavrest-dwbg_in/2], wcs, db)
    rslice = wavs2slice([wavrest+dwbg_in/2, wavrest+dwbg_out/2], wcs, db)
    # extract backgrounds on blue and red sides
    bgblu = data[:, bslice].mean(axis=1)
    bgred = data[:, rslice].mean(axis=1)
    # take weighted average, accounting for cases where the bg region
    # does not fit in the image
    weight_blu = data[:, bslice].size
    weight_red = data[:, rslice].size
    print('Background weights:', weight_blu, weight_red)
    bg = (bgblu*weight_blu + bgred*weight_red)/(weight_blu + weight_red)
    bgdata = np.zeros_like(data)
    bgdata += bg[:, None]
    return data - bgdata, bgdata
