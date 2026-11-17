#!/usr/bin/env python3
"""
Generates the two illustrative figures of the paper:

  1. pareto_nonconvex.pdf        - weighted sum cannot reach non-convex regions
  2. contrail_field_smoothing.pdf - Gaussian smoothing of the contrail field
                                    and its B-spline / linear interpolation
  3. mass_error.pdf              - error statistics of the fixed-mass assumption
                                    (compact replacement for Table I)
  4. flights_selection.pdf       - funnel of the flight-selection pipeline

Both are drawn at their final printed size (one IEEE column = 3.5 in) with
text sizes close to the body text, so include them WITHOUT rescaling:

    \\includegraphics[width=\\linewidth]{pareto_nonconvex.pdf}

Requirements:  numpy, scipy, matplotlib
Usage:         python make_figures.py [output_dir]      (default: ./figures)

Font: the figures use a Times-compatible font. Install one of
'Liberation Serif', 'TeX Gyre Termes' or 'Times New Roman' for the result to
match the paper text; otherwise matplotlib falls back to DejaVu Serif, which
is wider and will change the layout.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------- style ----
SERIF_FONTS = ['Liberation Serif', 'TeX Gyre Termes', 'Times New Roman',
               'DejaVu Serif']

# Colour palette shared by all figures: blue / red / rose, as in the
# Pareto-front figure. No orange, green or other hues.
C_DEEP, C_MID = '#0C447C', '#185FA5'      # blues: points / main curves
C_RED, C_RED_L = '#A32D2D', '#E24B4A'     # reds : non-supported / warnings
C_PURPLE = '#5C3D87'                      # purple: third series (distinct from
                                           # the blue/orange/green/dark-yellow
                                           # used to code methods in the results)
C_GREY = '#888780'                        # neutral: connecting lines only


def set_style():
    available = {f.name for f in font_manager.fontManager.ttflist}
    if not any(f in available for f in SERIF_FONTS[:3]):
        print('WARNING: no Times-compatible font found; falling back to '
              'DejaVu Serif (layout will differ).')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': SERIF_FONTS,
        'mathtext.fontset': 'stix',
        # sizes close to the IEEE body text (10 pt)
        'font.size': 10,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'text.usetex': False,
        # resolution and transparent background
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.facecolor': 'none',
        'axes.facecolor': 'none',
        'savefig.facecolor': 'none',
        'savefig.edgecolor': 'none',
        'savefig.transparent': True,
        # thin dashed light-grey grid, used consistently
        'grid.color': '#CCCCCC',
        'grid.linestyle': '--',
        'grid.linewidth': 0.4,
        'grid.alpha': 1.0,
        'axes.linewidth': 0.5,
    })


def style_axes(ax):
    """Grid below the data, no top/right spines."""
    ax.grid(True)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


# ------------------------------------------------------ figure 1: Pareto ----
def lower_hull(xs, ys):
    """Lower convex hull (Andrew's monotone chain). Returns an (m, 2) array."""
    hull = []
    for p in sorted(zip(xs, ys)):
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            cross = (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return np.array(hull)


def fig_pareto_nonconvex(out_path):
    # Synthetic front: convex arms with a concave (non-supported) bulge
    fuel = np.array([1000, 1020, 1045, 1070, 1095, 1120,
                     1145, 1170, 1195, 1220, 1250, 1290], dtype=float)
    # Peaks at 80 (was 100) so the y axis can stop at 83 and the figure stay short;
    # the non-convex bulge (the 16 * Gaussian) is unchanged.
    contrail = (4 + 76 * np.exp(-(fuel - 1000) / 70)
                + 16 * np.exp(-((fuel - 1145) / 45) ** 2))

    hull = lower_hull(fuel, contrail)
    hx, hy = hull[:, 0], hull[:, 1]
    supported = np.array([np.any(np.isclose(hx, f) & np.isclose(hy, c))
                          for f, c in zip(fuel, contrail)])
    nonsup = ~supported

    # 2.5 in tall (was 3.3) to save vertical space in the column.
    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    # hull vertices bracketing the non-supported points
    lo, hi = fuel[nonsup].min(), fuel[nonsup].max()
    xL = hx[hx < lo].max()
    yL = hy[np.isclose(hx, xL)][0]
    xR = hx[hx > hi].min()
    yR = hy[np.isclose(hx, xR)][0]

    # region unreachable by any weighted sum
    seg = (fuel >= xL) & (fuel <= xR)
    chord = np.interp(fuel[seg], [xL, xR], [yL, yR])
    ax.fill_between(fuel[seg], chord, contrail[seg], color=C_RED_L, alpha=0.18,
                    zorder=1, label='Unreachable by weighted sum')

    # true Pareto front
    ax.plot(fuel, contrail, color=C_MID, lw=1.3, zorder=3,
            label='True Pareto front')

    # weighted-sum level sets (slope = chord slope, i.e. tangent to the hull)
    slope = (yR - yL) / (xR - xL)
    tx = np.linspace(xL - 32, xR + 32, 50)
    ty = yL + slope * (tx - xL)
    ax.plot(tx, ty, color=C_RED, lw=1.1, zorder=4)
    for offset in (22, 44):
        ax.plot(tx, ty + offset, color=C_RED, lw=0.75, ls=':', alpha=0.5,
                zorder=2)

    # epsilon-constraint budget
    eps_x = 1145.0
    ax.axvline(eps_x, color='black', lw=1.1, ls='--', alpha=0.9, zorder=2)
    ax.text(eps_x + 5, 42, r'$\epsilon$', fontsize=10, ha='left', va='center')

    # solutions
    ax.scatter(fuel[supported], contrail[supported], s=20, color=C_DEEP,
               zorder=6, edgecolors='white', linewidths=0.5)
    ax.scatter(fuel[nonsup], contrail[nonsup], s=20, color=C_RED, marker='D',
               zorder=6, edgecolors='white', linewidths=0.5)

    # annotations
    arrow = dict(arrowstyle='->', color=C_RED, lw=0.7)
    ax.annotate('Weighted-sum level set\n(slope set by $c$)',
                xy=(xR - 12, yL + slope * (xR - 12 - xL)), xytext=(1232, 42),
                fontsize=8, color=C_RED, ha='center', arrowprops=arrow)
    ax.annotate('Non-convex region:\noptimal for no $c$',
                xy=(1145, contrail[fuel == 1145][0]), xytext=(1050, 9),
                fontsize=8, color=C_RED, ha='center', arrowprops=arrow)

    ax.set_xlabel('Fuel cost (kg)')
    ax.set_ylabel('Contrail cost')
    ax.set_xlim(985, 1310)
    ax.set_ylim(-4, 83)
    ax.set_yticks([0, 20, 40, 60, 80])
    style_axes(ax)
    ax.legend(loc='upper right', framealpha=1.0, facecolor='white',
              edgecolor='#CCCCCC', handlelength=1.6, borderpad=0.45,
              labelspacing=0.4)

    fig.tight_layout(pad=0)
    fig.savefig(out_path, format='pdf', bbox_inches='tight', pad_inches=0.02,
                transparent=True)
    plt.close(fig)
    print(f'Saved {out_path}')


# ------------------------------------------- figure 2: contrail smoothing ----
def fig_contrail_smoothing(out_path):
    n = 9                                   # grid size (n x n)
    sigmas = (0, 1, 2)                      # Gaussian filter widths [cells]
    slice_row = 4                           # cross-section used below

    # Synthetic binary contrail field and its smoothed versions
    binary = np.zeros((n, n))
    binary[3:6, 3:6] = 1
    fields = {s: binary if s == 0 else gaussian_filter(binary, sigma=s)
              for s in sigmas}

    cmap = LinearSegmentedColormap.from_list(
        'contrail', ['#F3E9EB', '#B5D4F4', '#378ADD', '#185FA5', '#0C447C'])

    fig = plt.figure(figsize=(3.5, 2.55))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.045],
                  height_ratios=[1, 1.05], hspace=0.14, wspace=0.30)

    # ---- top row: fields -------------------------------------------------
    for i, s in enumerate(sigmas):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(fields[s], cmap=cmap, vmin=0, vmax=1,
                       interpolation='nearest', aspect='equal')
        for edge in (slice_row - 0.5, slice_row + 0.5):     # cross-section
            ax.axhline(edge, color=C_RED_L, lw=0.9, ls='--', alpha=0.85)
        ax.set_title(f'$\\sigma = {s}$', pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel('Altitude levels')

    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, 0.5, 1])
    cb.outline.set_linewidth(0.5)
    cax.set_title('Contrail\ncost', fontsize=9, pad=4)

    # ---- bottom row: interpolation along the cross-section -----------------
    cells = np.arange(n)
    xf = np.linspace(0, n - 1, 500)
    cubic, linear = {}, {}
    for s in sigmas:
        row = fields[s][slice_row]
        cubic[s] = make_interp_spline(cells, row, k=3)(xf)   # cubic B-spline
        linear[s] = np.interp(xf, cells, row)

    y_min = min(min(c.min() for c in cubic.values()), -0.05) - 0.10
    y_max = 1.68               # headroom: legend sits above the sigma=2 curve

    handles = None
    for i, s in enumerate(sigmas):
        ax = fig.add_subplot(gs[1, i])
        row, yb, yl = fields[s][slice_row], cubic[s], linear[s]

        ax.axhline(0, color=C_GREY, lw=0.5)
        ax.axhline(1, color=C_GREY, lw=0.5, ls='--', alpha=0.5)

        # out-of-range excursions of the B-spline, shaded and annotated
        for mask, pick, ref, dy in ((yb < 0, np.argmin, 0, -0.03),
                                    (yb > 1, np.argmax, 1, +0.03)):
            if np.any(mask):
                ax.fill_between(xf, yb, ref, where=mask, color=C_RED_L,
                                alpha=0.35, label='Non-physical values')
                k = pick(yb)
                ax.annotate(f'{yb[k]:.2f}', xy=(xf[k], yb[k]),
                            xytext=(xf[k] + 0.7, yb[k] + dy), fontsize=8,
                            color=C_RED,
                            arrowprops=dict(arrowstyle='->', color=C_RED,
                                            lw=0.6))

        ax.plot(xf, yl, color=C_PURPLE, lw=1.0, zorder=2, label='Linear')
        ax.plot(xf, yb, color=C_MID, lw=1.1, zorder=3, label='B-spline')
        ax.scatter(cells, row, s=8, color=C_DEEP, zorder=4,
                   edgecolors='white', linewidths=0.4, label='Grid values')

        ax.set_xlim(-0.3, n - 0.7)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_yticks([0, 0.5, 1])
        style_axes(ax)
        if i == 0:
            ax.set_ylabel('Contrail cost')
            handles = dict(zip(*reversed(ax.get_legend_handles_labels())))
        else:
            ax.set_yticklabels([])

    fig.text(0.46, 0.02, 'Waypoint index', ha='center', va='bottom')

    # Legend in the emptiest panel (sigma = 2, bottom right); `ax` is that axes
    order = ['B-spline', 'Linear', 'Grid values', 'Non-physical values']
    leg = ax.legend([handles[k] for k in order], order, loc='upper right',
                    bbox_to_anchor=(1.50, 1.03), fontsize=8, frameon=True,
                    facecolor='white', framealpha=1.0, edgecolor='#CCCCCC',
                    handlelength=1.1, handletextpad=0.4, labelspacing=0.25,
                    borderpad=0.35)
    leg.get_frame().set_linewidth(0.5)

    fig.subplots_adjust(left=0.13, right=0.90, top=0.92, bottom=0.18)
    fig.savefig(out_path, format='pdf', bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f'Saved {out_path}')
    for s in sigmas:
        print(f'  sigma={s}: B-spline range '
              f'[{cubic[s].min():.3f}, {cubic[s].max():.3f}]')

# --------------------------------------- figure 3: flight-selection funnel ----
def fig_flights_selection(out_path):
    """Kept/dropped funnel for the flight-selection pipeline.

    Numbers match the September 2022-02-20 campaign, revision 29 Sep 2026
    (14,000 -> 13,917 -> 5,919 -> 5,385 -> 3,761 final population: 3,097
    narrow-body at m0=0.80 + 664 wide-body at m0=0.60). Re-check against the
    campaign source before reusing this figure for a later revision.
    """
    steps = [
        ('OpenSky data (EU, 20 Feb 2022)', 14000, None),
        ('Typecode found in aircraft database',              13917, 83),
        ('Persistent contrail',                 5919, 7998),
        ('Covered with OpenAP and ERA5 Data',                                        5385, 534),
        ('Converged (final population)',                        3761, 1624),
    ]

    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    n = len(steps)
    ys = np.arange(n)[::-1] * 0.85
    kept_max = steps[0][1]
    bar_h = 0.42

    for y, (label, kept, dropped) in zip(ys, steps):
        frac = kept / kept_max
        ax.barh(y, frac, height=bar_h, color=C_DEEP, zorder=3,
                edgecolor='none')
        if dropped:
            ax.barh(y, 1 - frac, left=frac, height=bar_h, color='#E6DEF0',
                    edgecolor=C_RED, linewidth=0.6, zorder=2)
            ax.text(1.28, y, f'\u2212{dropped:,}', ha='right', va='center',
                    fontsize=6, color=C_RED)
        ax.text(frac - 0.02, y, f'{kept:,}', ha='right', va='center',
                fontsize=6, color='white', zorder=4)
        ax.text(0.0, y + bar_h / 2 + 0.06, label, ha='left', va='bottom',
                fontsize=8)

    ax.set_xlim(-0.01, 1.30)
    ax.set_ylim(-0.3, ys[0] + bar_h + 0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.grid(False)

    fig.tight_layout(pad=0)
    fig.savefig(out_path, format='pdf', bbox_inches='tight',
                pad_inches=0.05, transparent=True)
    # fig.savefig(out_path, format='pdf', bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f'Saved {out_path}')

# ----------------------------------------------------------------- main ----
if __name__ == '__main__':
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'figures')
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    fig_pareto_nonconvex(out_dir / 'pareto_nonconvex.pdf')
    fig_contrail_smoothing(out_dir / 'contrail_field_smoothing.pdf')
    fig_flights_selection(out_dir / 'flights_selection.pdf')