from astropy.io import fits
import aplpy
from matplotlib import pyplot as plt
from matplotlib.figure import SubplotParams

figfile = 'owl-isovel-figs.png'

def chan2vrange(ichan, v0=5.0, dv=10.0):
    """Convert integer channel number to velocity range"""
    v1 = v0 + (ichan - 0.5)*dv
    v2 = v0 + (ichan + 0.5)*dv
    return '{:+04d}{:+04d}'.format(int(v1), int(v2))

subplotpars = SubplotParams(left=0.0, right=1.0,
                            bottom=0.0, top=1.0,
                            wspace=0.0, hspace=0.0)
fig = plt.figure(figsize=(26, 4), subplotpars=subplotpars)

ichannels = range(-6, 7)
lineid = 'ha'

linemax = {'ha': 0.12, 'nii': 0.08}

n = 0
for j, lineid in enumerate(['ha', 'nii']):
    for i, ichan in enumerate(ichannels):
        n += 1
        fitsfile = 'owl-dslits-{}{}-multibin.fits'.format(lineid, chan2vrange(ichan))
        f = aplpy.FITSFigure(fitsfile, figure=fig, subplot=(2, 13, n))
        f.show_grayscale(vmin=0.0, vmax=linemax[lineid], invert=True, interpolation='none')
        f.ticks.set_color('black')
        f.hide_axis_labels()
        f.hide_tick_labels()

fig.savefig(figfile, dpi=150)
print(figfile)
