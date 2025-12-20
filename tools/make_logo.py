import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import rcParams

# Output paths (work both if publish/ is repo root or subdir)
BASE = Path('publish') if Path('publish/code').exists() else Path('.')
OUT_SQUARE = BASE / 'logo.png'
OUT_BANNER = BASE / 'logo_banner.png'


def draw_square_logo(path: Path, brand_main: str = "EurekaMesh", brand_sub: str = "Labs"):
    fig = plt.figure(figsize=(4, 4), dpi=256)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Background
    ax.set_facecolor('#0b1220')  # deep navy

    # Symbol: mesh of circles (abstract "mesh")
    colors = ['#60a5fa', '#34d399', '#a78bfa']
    positions = [(0.35, 0.65), (0.65, 0.65), (0.50, 0.40)]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    for (x, y), c in zip(positions, colors):
        circ = Circle((x, y), 0.16, facecolor=c, alpha=0.9)
        ax.add_patch(circ)
    # connecting lines
    ax.plot([0.35, 0.65], [0.65, 0.65], color='#93c5fd', lw=3, alpha=0.9)
    ax.plot([0.35, 0.50], [0.65, 0.40], color='#6ee7b7', lw=3, alpha=0.9)
    ax.plot([0.65, 0.50], [0.65, 0.40], color='#c4b5fd', lw=3, alpha=0.9)

    # Brand text
    rcParams['font.family'] = 'DejaVu Sans'
    ax.text(0.5, 0.14, brand_main, ha='center', va='center', color='white',
            fontsize=26, weight='bold', alpha=0.98)
    ax.text(0.5, 0.07, brand_sub, ha='center', va='center', color='#cbd5e1',
            fontsize=14, alpha=0.9)

    fig.savefig(path, facecolor=ax.get_facecolor(), edgecolor='none')
    plt.close(fig)


def draw_banner_logo(path: Path, brand_main: str = "EurekaMesh", brand_sub: str = "Labs"):
    fig = plt.figure(figsize=(6, 2), dpi=256)
    fig.patch.set_facecolor('#0b1220')

    # Left icon as a square inset axes to avoid distortion
    icon_ax = fig.add_axes([0.06, 0.15, 0.24, 0.70])  # x, y, w, h
    icon_ax.set_axis_off()
    icon_ax.set_facecolor('#0b1220')
    icon_ax.set_xlim(0, 1)
    icon_ax.set_ylim(0, 1)
    icon_ax.set_aspect('equal', adjustable='box')

    colors = ['#60a5fa', '#34d399', '#a78bfa']
    positions = [(0.30, 0.65), (0.70, 0.65), (0.50, 0.35)]
    for (x, y), c in zip(positions, colors):
        circ = Circle((x, y), 0.20, facecolor=c, alpha=0.9)
        icon_ax.add_patch(circ)
    icon_ax.plot([0.30, 0.70], [0.65, 0.65], color='#93c5fd', lw=3, alpha=0.9)
    icon_ax.plot([0.30, 0.50], [0.65, 0.35], color='#6ee7b7', lw=3, alpha=0.9)
    icon_ax.plot([0.70, 0.50], [0.65, 0.35], color='#c4b5fd', lw=3, alpha=0.9)

    # Text placed at figure level to keep proportions on any aspect ratio
    rcParams['font.family'] = 'DejaVu Sans'
    fig.text(0.36, 0.60, brand_main, ha='left', va='center', color='white',
             fontsize=34, weight='bold')
    fig.text(0.36, 0.38, brand_sub, ha='left', va='center', color='#cbd5e1',
             fontsize=18)

    fig.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


def main():
    OUT_SQUARE.parent.mkdir(parents=True, exist_ok=True)
    draw_square_logo(OUT_SQUARE)
    draw_banner_logo(OUT_BANNER)
    print(f"Saved {OUT_SQUARE} and {OUT_BANNER}")


if __name__ == '__main__':
    main()


