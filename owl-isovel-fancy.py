from astropy.io import fits
import aplpy
from matplotlib import pyplot as plt
from matplotlib.figure import SubplotParams

figfile = 'owl-isovel-fancy.png'

def chan2vrange(ichan, v0=5.0, dv=10.0):
    """Convert integer channel number to velocity range"""
    v1 = v0 + (ichan - 0.5)*dv
    v2 = v0 + (ichan + 0.5)*dv
    return '{:+04d}{:+04d}'.format(int(v1), int(v2))

# Layout dimensions, all in inches
paper_size = 11, 11
margin = 0.5
stamp_size = 1.4  # inches

# Origin of each of the three major sections of the figure
origin = {
    'nii': (margin, margin),
    'ha': (margin, margin + 2*stamp_size + margin),
    'allvels': (margin + 0.75*stamp_size,
                paper_size[1] - 2*margin - 2*stamp_size),
}
# (x, y) offsets for the V channels in units of stamp_size
snake = {
    # negative V channels in top row
    -6: (0, 1),
    -5: (1, 1),
    -4: (2, 1),
    -3: (3, 1),
    -2: (4, 1),
    -1: (5, 1),
    # central panel on far-right, half-way down
    0: (6, 0.5),
    # positive V channels in bottom row
    6: (0, 0),
    5: (1, 0),
    4: (2, 0),
    3: (3, 0),
    2: (4, 0),
    1: (5, 0),
}

fig = plt.figure(figsize=paper_size)

ichannels = range(-6, 7)
linemax = {'ha': 0.12, 'nii': 0.08}

ra_tick_spacing = 10.0*15/(60*60)  # 10 m
dec_tick_spacing = 60/(60*60)      # 60''
xstar, ystar = 168.69878, 55.019228

# A snakey sequence of channel maps for each line
for lineid in ['ha', 'nii']:
    for ichan in ichannels:
        x0 = origin[lineid][0] + snake[ichan][0]*stamp_size
        y0 = origin[lineid][1] + snake[ichan][1]*stamp_size
        # Define window [x, y, w, h] in fractions of figure size
        window = [x0/paper_size[0], y0/paper_size[1],
                  stamp_size/paper_size[0], stamp_size/paper_size[1]]
        fitsfile = 'owl-dslits-{}{}-multibin.fits'.format(lineid, chan2vrange(ichan))
        f = aplpy.FITSFigure(fitsfile, figure=fig, subplot=window)
        f.show_colorscale(vmin=0.0, vmax=linemax[lineid],
                         cmap='viridis', interpolation='none')
        f.ticks.set_color('gray')
        f.frame.set_color('gray')
        f.hide_axis_labels()
        f.hide_tick_labels()
        f.ticks.set_xspacing(ra_tick_spacing)
        f.ticks.set_yspacing(dec_tick_spacing)
        f.show_markers([xstar], [ystar], marker='*')
#
# And the four all-vels images across the top
#

rowA = [['imslit-median-sub', 1.5, 0],
        ['owl-dslits-ha-allvels-multibin', 1.0, 0],
        ['owl-dslits-nii-allvels-multibin', 0.4, 0],
        ['imslit-median-sub', 1.5, 0]]

rowB = [['owl-slits-ha-allvels', 0.85, 3],
        ['owl-dslits-ha-allvels-bin004', 1.0, 0],
        ['owl-dslits-ha-allvels-bin016', 1.0, 0],
        ['owl-dslits-ha-allvels-bin064', 1.0, 0]]

for j, row in enumerate([rowA, rowB]):
    for i, [fn, vmax, ihdu] in enumerate(row):
        x0 = origin['allvels'][0] + i*(5./4.)*stamp_size
        y0 = origin['allvels'][1] + j*1.4*stamp_size
        window = [x0/paper_size[0], y0/paper_size[1],
                  stamp_size/paper_size[0], stamp_size/paper_size[1]]
        fitsfile = fn  + '.fits'
        f = aplpy.FITSFigure(fitsfile, figure=fig, subplot=window, hdu=ihdu)
        f.show_grayscale(vmin=0.0, vmax=vmax, invert=True, interpolation='none')
        f.ticks.set_color('gray')
        if i > 0 or j > 0:
            f.hide_axis_labels()
            f.hide_tick_labels()
        else:
            f.axis_labels.set_font(size='x-small')
            f.tick_labels.set_font(size='x-small')
            f.tick_labels.set_xformat('hh:mm:ss.s')
            f.tick_labels.set_yformat('dd:mm:ss.s')
            f.tick_labels.set_style('colons')
        f.ticks.set_xspacing(ra_tick_spacing)
        f.ticks.set_yspacing(dec_tick_spacing)
        f.show_markers([xstar], [ystar], marker='*')

fig.savefig(figfile, dpi=150)
print(figfile, end='')
