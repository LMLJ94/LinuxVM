"""Box plots of the hand-made features per class (training set), from the
feature cache that evaluate_rf.py writes. Run evaluate_rf.py first."""
import os

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from features import FEATURE_NAMES
from evaluate_rf import CACHE_PATH, RECIPE

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'pytorch-quickstart', 'figures', 'rf_feature_boxplots.png')

CLASSES = ['Healthy', 'Powdery', 'Rust']

# (feature, panel title, y-scale) — the six most important features in the
# colour + texture Random Forest, in order of importance.
PANELS = [
    ('low_saturation_fraction', 'Andel grålige pixels\n(low_saturation_fraction)', 'linear'),
    ('hue_hist_7', 'Andel rødlige pixels, hue 157–180\n(hue_hist_7, symlog-skala)', 'symlog'),
    ('green_fraction', 'Andel grønne pixels\n(green_fraction)', 'linear'),
    ('laplacian_variance', 'Laplace-varians\n(laplacian_variance, log-skala)', 'log'),
    ('rust_fraction', 'Andel orange/brune pixels\n(rust_fraction, symlog-skala)', 'symlog'),
    ('rust_spot_count', 'Antal rustpletter\n(rust_spot_count, symlog-skala)', 'symlog'),
]

FILL = '#2a78d6'
SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e4e3df'


def danish_number(x):
    # Two significant digits, no scientific notation, decimal comma.
    if x == 0:
        return '0'
    if abs(x) >= 10:
        return f'{x:.0f}'
    return np.format_float_positional(x, precision=2, fractional=False, trim='-').replace('.', ',')


def main():
    cache = joblib.load(CACHE_PATH)
    if cache.get('recipe') != RECIPE:
        raise RuntimeError(f'{CACHE_PATH} was made with a different feature recipe; re-run evaluate_rf.py')
    train = cache['splits']['train']

    plt.rcParams.update({'font.size': 10, 'text.color': TEXT, 'axes.labelcolor': TEXT_SECONDARY,
                         'xtick.color': TEXT_SECONDARY, 'ytick.color': TEXT_SECONDARY})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), facecolor=SURFACE)

    for ax, (name, title, scale) in zip(axes.flat, PANELS):
        col = FEATURE_NAMES.index(name)
        data = [np.array([s[2][col] for s in train if s[1] == c]) for c in CLASSES]

        bp = ax.boxplot(data, widths=0.55, patch_artist=True, whis=(5, 95), showfliers=False,
                        medianprops={'color': TEXT, 'linewidth': 2},
                        whiskerprops={'color': TEXT_SECONDARY, 'linewidth': 1},
                        capprops={'color': TEXT_SECONDARY, 'linewidth': 1},
                        boxprops={'edgecolor': SURFACE, 'linewidth': 2})
        for box in bp['boxes']:
            box.set_facecolor(FILL)
            box.set_alpha(0.85)

        if scale == 'log':
            ax.set_yscale('log')
        elif scale == 'symlog':
            ax.set_yscale('symlog', linthresh=1 if name == 'rust_spot_count' else 1e-3)
            ax.set_ylim(bottom=0)

        for i, d in enumerate(data, start=1):
            med = np.median(d)
            label = danish_number(med)
            ax.annotate(label, (i + 0.3, med), xytext=(3, 0), textcoords='offset points',
                        va='center', ha='left', fontsize=8.5, color=TEXT_SECONDARY)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: danish_number(v)))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
        if scale == 'log':
            ax.set_yticks([300, 500, 1000, 2000, 4000])
        ax.set_xticks([1, 2, 3], CLASSES)
        ax.set_title(title, fontsize=10, color=TEXT, loc='left')
        ax.set_facecolor(SURFACE)
        ax.grid(axis='y', color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        for side in ['left', 'bottom']:
            ax.spines[side].set_color(GRID)
        ax.tick_params(length=0)

    fig.suptitle(f'Håndlavede features pr. klasse (træningssæt, n = {len(train)})',
                 fontsize=13, x=0.01, ha='left', color=TEXT)
    fig.text(0.01, 0.005, 'Boks: 25.–75. percentil, streg: median (værdi til højre), whiskers: 5.–95. percentil. '
             'Features beregnet efter nedskalering til 256×256 px.', fontsize=8.5, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(OUTPUT_PATH, dpi=150, facecolor=SURFACE)
    print(f'Saved {os.path.abspath(OUTPUT_PATH)}')


if __name__ == '__main__':
    main()
