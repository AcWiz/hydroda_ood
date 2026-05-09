"""HydroDA-OOD Dataset Splits Visualization (Nature Journal Style).

Creates a 4-panel figure showing:
1. US 6-region geographic distribution with cartopy map projection
2. Temporal split timeline (source_train / target_support / target_query)
3. LORO cross-validation structure
4. K-date support sampling strategy

Nature Style Guidelines:
- Color palette: Deep blue (#2c3e50), Light blue (#3498db), Red (#c0392b),
  Green (#27ae60), Gray (#7f8c8d, #95a5a6)
- Font: Arial/Helvetica (sans-serif), avoid Times New Roman
- DPI: 300+ for publication standard
- Figure size: Single column (~8.5cm) or double column (~17cm)
- Clean axes with minimal spines

Usage:
    python scripts/plot_dataset_splits.py --out-png figures/dataset_splits_overview.png
"""

import argparse
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Nature-style color palette
NATURE_COLORS = {
    "deep_blue": "#2c3e50",
    "light_blue": "#3498db",
    "red": "#c0392b",
    "green": "#27ae60",
    "orange": "#e67e22",
    "purple": "#8e44ad",
    "gray_dark": "#7f8c8d",
    "gray_light": "#95a5a6",
    "background": "#fafafa",
}

# Region colors (Nature-style muted palette, colorblind-friendly)
# Indices 0-6 map to: background, R1-R6
REGION_COLORS = [
    "#ffffff",  # 0: background (white, transparent)
    "#ef9a9a",  # R1: light red
    "#ffcc80",  # R2: light orange
    "#a5d6a7",  # R3: light green
    "#90caf9",  # R4: light blue
    "#ce93d8",  # R5: light purple
    "#9fa8da",  # R6: light indigo
]


def configure_matplotlib_style():
    """Configure matplotlib for Nature journal style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def plot_us_regions_map(ax, masks_nc, latlon_nc):
    """Panel A: US region masks geographic distribution with cartopy."""
    ds_masks = xr.open_dataset(masks_nc)
    ds_geo = xr.open_dataset(latlon_nc)

    mask_int = ds_masks["region_mask_integer"].values
    lat = ds_geo["latitude"].values
    lon = ds_geo["longitude"].values

    region_names = {
        1: "R1",
        2: "R2",
        3: "R3",
        4: "R4",
        5: "R5",
        6: "R6",
    }

    region_full_names = {
        1: "Southwest Desert",
        2: "S. Great Plains",
        3: "CA Central Valley",
        4: "Corn Belt",
        5: "Southeast US",
        6: "Central Rockies",
    }

    # Set up map projection
    projection = ccrs.PlateCarree()
    ax.set_extent([-125, -66, 24, 50], crs=projection)

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor="#fafafa", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=NATURE_COLORS["gray_dark"])
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor=NATURE_COLORS["gray_light"])

    # Plot region masks - use integer indices directly without vmin/vmax
    cmap = plt.cm.colors.ListedColormap(REGION_COLORS)
    im = ax.pcolormesh(
        lon, lat, mask_int,
        cmap=cmap,
        shading="auto",
        transform=projection,
        zorder=1,
    )

    # Add region labels at approximate centers
    region_centers = {
        1: (-112.5, 34.0),
        2: (-100.0, 35.5),
        3: (-120.5, 37.5),
        4: (-90.0, 42.0),
        5: (-85.0, 32.5),
        6: (-108.5, 41.5),
    }
    for rid, (cx, cy) in region_centers.items():
        ax.text(
            cx, cy,
            f"{region_names[rid]}\n{region_full_names[rid]}",
            fontsize=6,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            transform=projection,
            zorder=3,
        )

    # Add gridlines
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color=NATURE_COLORS["gray_light"],
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7, "color": NATURE_COLORS["gray_dark"]}
    gl.ylabel_style = {"size": 7, "color": NATURE_COLORS["gray_dark"]}

    # Panel label
    ax.text(
        -0.05, 1.05, "A",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="right",
        color=NATURE_COLORS["deep_blue"],
    )

    ax.set_title("US Hydroclimatic Regions", fontsize=11, fontweight="bold", pad=10)

    ds_masks.close()
    ds_geo.close()


def plot_temporal_timeline(ax, splits_json):
    """Panel B: Temporal split timeline (Nature style)."""
    import json

    with open(splits_json) as f:
        data = json.load(f)

    first = data["splits"][0]
    source_start, source_end = first["source_train_period"].split(" to ")
    support_start, support_end = first["target_support_period"].split(" to ")
    query_start, query_end = first["target_query_period"].split(" to ")

    # Timeline configuration
    ax.set_xlim(2014.5, 2026)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(2015, 2026))

    # Source train period (2015-2020)
    ax.axvspan(2015, 2020.5, ymin=0.35, ymax=0.65, color=NATURE_COLORS["deep_blue"], alpha=0.6)
    ax.text(2017.5, 0.7, "Source Training", ha="center", va="center", fontsize=8,
            color=NATURE_COLORS["deep_blue"], fontweight="bold")
    ax.text(2017.5, 0.55, "2015–2020", ha="center", va="center", fontsize=7,
            color=NATURE_COLORS["deep_blue"])

    # Target support period (2021)
    ax.axvspan(2021, 2021.5, ymin=0.35, ymax=0.65, color=NATURE_COLORS["orange"], alpha=0.8)
    ax.text(2021.25, 0.7, "Target Support", ha="center", va="center", fontsize=8,
            color=NATURE_COLORS["orange"], fontweight="bold")
    ax.text(2021.25, 0.55, "2021", ha="center", va="center", fontsize=7,
            color=NATURE_COLORS["orange"])

    # Target query period (2022-2025)
    ax.axvspan(2022, 2025.5, ymin=0.35, ymax=0.65, color=NATURE_COLORS["red"], alpha=0.6)
    ax.text(2023.5, 0.7, "Target Query", ha="center", va="center", fontsize=8,
            color=NATURE_COLORS["red"], fontweight="bold")
    ax.text(2023.5, 0.55, "2022–2025", ha="center", va="center", fontsize=7,
            color=NATURE_COLORS["red"])

    # K values annotation
    ax.text(2021.25, 0.2, "K ∈ {0, 4, 12, 24}", ha="center", va="center", fontsize=8,
            color=NATURE_COLORS["gray_dark"], style="italic")

    # Year boundary lines
    for y in [2021, 2022]:
        ax.axvline(x=y, color=NATURE_COLORS["gray_light"], linestyle="--", alpha=0.7, linewidth=0.8)

    # Clean up axes
    ax.set_xlabel("Year", fontsize=9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=7)

    # Panel label
    ax.text(
        -0.05, 1.05, "B",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="right",
        color=NATURE_COLORS["deep_blue"],
    )

    ax.set_title("Temporal Split Timeline", fontsize=11, fontweight="bold", pad=10)


def plot_loro_structure(ax, splits_json):
    """Panel C: LORO cross-validation structure (Nature style)."""
    import json

    with open(splits_json) as f:
        data = json.load(f)

    regions = ["R1", "R2", "R3", "R4", "R5", "R6"]
    n_regions = len(regions)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, n_regions + 0.5)
    ax.set_aspect(0.8)

    for i, region_id in enumerate(regions):
        y = n_regions - i

        # Target region marker (red box)
        ax.add_patch(
            plt.Rectangle((0.3, y - 0.15), 1.2, 0.3, color=NATURE_COLORS["red"], alpha=0.85)
        )
        ax.text(0.9, y, f"US-{region_id}", ha="center", va="center", fontsize=7,
                color="white", fontweight="bold")

        # Arrow indicating training
        ax.annotate(
            "", xy=(4.0, y), xytext=(2.0, y),
            arrowprops=dict(arrowstyle="->", color=NATURE_COLORS["gray_dark"], lw=1.2)
        )

        # Source regions (5 circles in blue)
        source_regions = [r for r in regions if r != region_id]
        for j, src in enumerate(source_regions):
            x = 4.5 + j * 0.95
            ax.add_patch(
                plt.Circle((x, y), 0.18, color=NATURE_COLORS["light_blue"], alpha=0.7)
            )
            ax.text(x, y, src, ha="center", va="center", fontsize=5, color="white",
                    fontweight="bold")

    ax.set_title("LORO Cross-Validation (6 folds)", fontsize=11, fontweight="bold", pad=10)
    ax.axis("off")

    # Clean legend
    target_patch = mpatches.Patch(color=NATURE_COLORS["red"], label="Target (held out)")
    source_patch = mpatches.Patch(color=NATURE_COLORS["light_blue"], alpha=0.7, label="Source (5 regions)")
    ax.legend(
        handles=[target_patch, source_patch],
        loc="lower right",
        fontsize=7,
        frameon=True,
        fancybox=False,
        edgecolor=NATURE_COLORS["gray_light"],
    )

    # Panel label
    ax.text(
        -0.05, 1.05, "C",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="right",
        color=NATURE_COLORS["deep_blue"],
    )


def plot_kdate_sampling(ax):
    """Panel D: K-date support sampling strategy (Nature style)."""
    months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    x = np.arange(len(months))

    ax.set_xlim(-0.5, 12)
    ax.set_ylim(0, 5)

    # Background month indicators
    for xi in x:
        ax.bar(xi, 0.2, bottom=0, color=NATURE_COLORS["gray_light"], alpha=0.2, width=0.8)

    # K=0: baseline reference
    ax.axhline(y=0.8, color=NATURE_COLORS["gray_dark"], linestyle=":", alpha=0.5, linewidth=0.8)
    ax.text(12.2, 0.8, "K=0", ha="left", va="center", fontsize=7,
            color=NATURE_COLORS["gray_dark"], style="italic")

    # K=4: one per quarter
    quarters = [0, 3, 6, 9]
    k4_y = 1.8
    for xi in quarters:
        ax.bar(xi + 0.5, 0.35, bottom=k4_y, color=NATURE_COLORS["deep_blue"], alpha=0.7, width=0.6)
    ax.text(12.2, k4_y + 0.17, "K=4", ha="left", va="center", fontsize=7,
            color=NATURE_COLORS["deep_blue"], fontweight="bold")

    # K=12: one per month
    k12_y = 2.8
    for xi in x:
        ax.bar(xi + 0.5, 0.3, bottom=k12_y, color=NATURE_COLORS["orange"], alpha=0.7, width=0.5)
    ax.text(12.2, k12_y + 0.15, "K=12", ha="left", va="center", fontsize=7,
            color=NATURE_COLORS["orange"], fontweight="bold")

    # K=24: two per month
    k24_y = 3.8
    for xi in x:
        ax.bar(xi + 0.35, 0.2, bottom=k24_y, color=NATURE_COLORS["green"], alpha=0.6, width=0.28)
        ax.bar(xi + 0.65, 0.2, bottom=k24_y + 0.25, color=NATURE_COLORS["green"], alpha=0.6, width=0.28)
    ax.text(12.2, k24_y + 0.35, "K=24", ha="left", va="center", fontsize=7,
            color=NATURE_COLORS["green"], fontweight="bold")

    ax.set_xticks(x + 0.5)
    ax.set_xticklabels(months, fontsize=7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=7)

    # Panel label
    ax.text(
        -0.05, 1.05, "D",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="right",
        color=NATURE_COLORS["deep_blue"],
    )

    ax.set_title("K-Date Support Sampling", fontsize=11, fontweight="bold", pad=10)


def main():
    parser = argparse.ArgumentParser(description="Generate dataset splits overview (Nature style)")
    parser.add_argument(
        "--masks-nc",
        default="artifacts/regions/US_region_masks.nc",
        help="Path to US_region_masks.nc",
    )
    parser.add_argument(
        "--latlon-nc",
        default="artifacts/geolocation/US_latlon.nc",
        help="Path to US_latlon.nc",
    )
    parser.add_argument(
        "--splits-json",
        default="artifacts/splits/US_loro_kdate_splits.json",
        help="Path to US_loro_kdate_splits.json",
    )
    parser.add_argument(
        "--out-png",
        default="figures/dataset_splits_overview.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI (default: 300 for publication)",
    )
    args = parser.parse_args()

    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)

    # Apply Nature style configuration
    configure_matplotlib_style()

    # Create figure with Nature-style dimensions (double column for journal)
    fig = plt.figure(figsize=(17, 11))

    # Create grid layout: map takes more space, others are smaller
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.2, 1],
        height_ratios=[1, 1],
        hspace=0.35,
        wspace=0.3,
        left=0.06,
        right=0.98,
        top=0.94,
        bottom=0.08,
    )

    # Panel A: US regions map (with cartopy)
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    plot_us_regions_map(ax_map, args.masks_nc, args.latlon_nc)

    # Panel B: Temporal timeline
    ax_timeline = fig.add_subplot(gs[0, 1])
    plot_temporal_timeline(ax_timeline, args.splits_json)

    # Panel C: LORO structure
    ax_loro = fig.add_subplot(gs[1, 0])
    plot_loro_structure(ax_loro, args.splits_json)

    # Panel D: K-date sampling
    ax_kdate = fig.add_subplot(gs[1, 1])
    plot_kdate_sampling(ax_kdate)

    # Main title
    fig.suptitle(
        "HydroDA-OOD: Dataset Split Strategy",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Save with high DPI
    plt.savefig(args.out_png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {args.out_png} (DPI: {args.dpi})")


if __name__ == "__main__":
    main()