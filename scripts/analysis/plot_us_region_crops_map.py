#!/usr/bin/env python3
"""Region Crop Visualization — Nature Journal Style.

Generates:
1. fig_region_map_US.png    — 6 regions on US map with Cartopy
2. fig_US-RX_detail.png (×6) — each region detail with RGB + stats

Usage:
    python scripts/plot_us_region_crops_map.py --out-dir figures/regions
"""

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
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
REGION_COLORS = [
    "#ffffff",  # 0: background (white)
    "#ef9a9a",  # R1: light red — dryland sparse vegetation
    "#ffcc80",  # R2: light orange — semi-arid transition
    "#a5d6a7",  # R3: light green — irrigated managed agriculture
    "#90caf9",  # R4: light blue — rainfed agriculture
    "#ce93d8",  # R5: light purple — humid high vegetation
    "#9fa8da",  # R6: light indigo — mountain cold terrain stress
]

REGION_NAMES = {
    1: "R1",
    2: "R2",
    3: "R3",
    4: "R4",
    5: "R5",
    6: "R6",
}

REGION_FULL_NAMES = {
    1: "Dryland Sparse Vegetation",
    2: "Semi-Arid Transition",
    3: "Irrigated Managed Agriculture",
    4: "Rainfed Agriculture",
    5: "Humid High Vegetation",
    6: "Mountain Cold Terrain Stress",
}

REGION_REGIMES = {
    1: "dryland_sparse_vegetation",
    2: "semi_arid_transition",
    3: "irrigated_managed_agriculture",
    4: "rainfed_agriculture",
    5: "humid_high_vegetation",
    6: "mountain_cold_terrain_stress",
}

# Approximate region centers for US map labels
REGION_CENTERS = {
    1: (-112.5, 34.0),
    2: (-100.0, 35.5),
    3: (-120.5, 37.5),
    4: (-90.0, 42.0),
    5: (-85.0, 32.5),
    6: (-108.5, 41.5),
}


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


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_full_us_mask(region_dirs, full_shape=(256, 640)):
    """Build a full US mask array with all region masks placed at correct positions.

    Args:
        region_dirs: dict mapping region_id -> Path to region directory
        full_shape: shape of full US grid (H, W)

    Returns:
        full_mask: numpy array of shape full_shape with region integer IDs
    """
    full_mask = np.zeros(full_shape, dtype=np.float32)

    for region_id in range(1, 7):
        region_dir = region_dirs.get(f"US-R{region_id}")
        if region_dir is None:
            continue

        mask_path = region_dir / "region_mask.pt"
        if not mask_path.exists():
            continue

        meta_path = region_dir / "metadata.json"
        if not meta_path.exists():
            continue

        meta = load_json(meta_path)

        # Get the crop position from resolved_index_bbox
        bbox = meta.get("resolved_index_bbox", {})
        y_start = bbox.get("y_start", 0)
        y_end = bbox.get("y_end", full_shape[0])
        x_start = bbox.get("x_start", 0)
        x_end = bbox.get("x_end", full_shape[1])

        # Load and place the region mask at correct position
        crop_mask = torch.load(mask_path).numpy()
        crop_h, crop_w = crop_mask.shape

        # Use actual crop dimensions to determine placement
        full_mask[y_start:y_start + crop_h, x_start:x_start + crop_w] = crop_mask * region_id

    return full_mask


def plot_us_region_overview_map(ax, region_dirs, latlon_nc):
    """Plot all 6 US regions on a Cartopy map.

    Args:
        ax: matplotlib axes with cartopy projection
        region_dirs: dict mapping region_id -> Path to region directory
        latlon_nc: Path to US_latlon.nc
    """
    ds_geo = xr.open_dataset(latlon_nc)
    lat = ds_geo["latitude"].values
    lon = ds_geo["longitude"].values

    projection = ccrs.PlateCarree()
    ax.set_extent([-125, -66, 24, 50], crs=projection)

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor="#fafafa", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=NATURE_COLORS["gray_dark"])
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor=NATURE_COLORS["gray_light"])

    # Build full US mask with all regions placed at correct positions
    full_mask = build_full_us_mask(region_dirs, full_shape=lat.shape)

    # Plot full mask using pcolormesh
    cmap = plt.cm.colors.ListedColormap(REGION_COLORS)
    im = ax.pcolormesh(
        lon, lat, full_mask,
        cmap=cmap,
        shading="auto",
        transform=projection,
        zorder=1,
        vmin=0,
        vmax=6,
    )

    # Add region labels at approximate centers
    for rid in range(1, 7):
        cx, cy = REGION_CENTERS[rid]
        ax.text(
            cx, cy,
            f"R{rid}",
            fontsize=8,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            transform=projection,
            zorder=3,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=NATURE_COLORS["deep_blue"], alpha=0.7),
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

    ax.set_title("HydroDA-OOD: US Hydroclimatic Regions", fontsize=11, fontweight="bold", pad=10)

    ds_geo.close()


def plot_region_detail(region_id, region_dir, year=2015):
    """Generate detail figure for a single region.

    Creates a multi-panel figure showing:
    - RGB composite of SM_surface, SM_rootzone, soil_temp
    - Target increment (ΔSM_surface, ΔSM_rootzone)
    - Mask and loss coverage
    - Statistics

    Args:
        region_id: Region ID string (e.g., "US-R1")
        region_dir: Path to region directory
        year: Year to visualize (default 2015)

    Returns:
        matplotlib figure
    """
    numeric_id = int(region_id.split("-R")[1])

    # Load metadata
    meta_path = region_dir / "metadata.json"
    meta = load_json(meta_path) if meta_path.exists() else {}

    # Load data
    try:
        inp = torch.load(region_dir / "da_full" / str(year) / "input.pt")
        tgt_inc = torch.load(region_dir / "da_full" / str(year) / "target_increment.pt")
        loss = torch.load(region_dir / "da_full" / str(year) / "loss_mask.pt")
        mask = torch.load(region_dir / "region_mask.pt").numpy()
    except Exception as e:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error loading data: {e}", ha='center', va='center')
        ax.axis('off')
        return fig

    # Take first time step
    t0 = inp[0].numpy()   # [12, H, W]
    t0_tgt = tgt_inc[0].numpy()  # [2, H, W]
    t0_loss = loss[0].numpy()   # [H, W]
    H, W = t0_loss.shape

    # Channel names for input [SM_surface, SM_rootzone, soil_temp, ...]
    ch_names = ["SM_surface", "SM_rootzone", "soil_temp", "soil_temp_longwave"]

    # Create figure: 3x3 grid
    fig, axs = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle(
        f"{region_id} — {REGION_FULL_NAMES[numeric_id]} ({REGION_REGIMES[numeric_id]})",
        fontsize=13, fontweight="bold", y=0.98
    )

    # Row 1: Mask, Loss Mask, RGB composite
    # Mask
    ax = axs[0, 0]
    im = ax.imshow(mask, cmap='tab20', vmin=0, vmax=20)
    ax.set_title(f"Region Mask\n{mask.shape}", fontsize=9)
    ax.axis('off')

    # Loss mask
    ax = axs[0, 1]
    im = ax.imshow(t0_loss.astype(float), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"Loss Mask (t=0)\nTrue ratio: {t0_loss.sum()/t0_loss.size:.3f}", fontsize=9)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.03)

    # RGB composite: SM_surface, SM_rootzone, soil_temp
    ax = axs[0, 2]
    rgb = np.stack([
        (t0[0] - t0[0].min()) / (t0[0].max() - t0[0].min() + 1e-8),  # R: SM_surface
        (t0[1] - t0[1].min()) / (t0[1].max() - t0[1].min() + 1e-8),  # G: SM_rootzone
        (t0[2] - t0[2].min()) / (t0[2].max() - t0[2].min() + 1e-8),  # B: soil_temp
    ], axis=-1)
    ax.imshow(rgb)
    ax.set_title("RGB: SM_surf / SM_root / T_soil", fontsize=9)
    ax.axis('off')

    # Row 2: Input channels
    for c in range(3):
        ax = axs[1, c]
        vmin, vmax = t0[c].min(), t0[c].max()
        im = ax.imshow(t0[c], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"Input ch{c}: {ch_names[c]}", fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.03)

    # Row 3: Target increments and stats
    # ΔSM_surface
    ax = axs[2, 0]
    vmax = np.abs(t0_tgt[0]).max()
    im = ax.imshow(t0_tgt[0], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title("ΔSM_surface (t=0)", fontsize=9)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.03)

    # ΔSM_rootzone
    ax = axs[2, 1]
    vmax = np.abs(t0_tgt[1]).max()
    im = ax.imshow(t0_tgt[1], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title("ΔSM_rootzone (t=0)", fontsize=9)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.03)

    # Statistics
    ax = axs[2, 2]
    ax.axis('off')
    ext = meta.get("resolved_latlon_extent", {})
    stats = (
        f"Region: {region_id}\n"
        f"Regime: {REGION_REGIMES[numeric_id]}\n"
        f"Crop shape: {meta.get('crop_shape', 'N/A')}\n"
        f"Year {year} time steps: {meta.get('years', {}).get(str(year), {}).get('time_steps', 'N/A')}\n"
        f"Lat: [{ext.get('lat_min', 'N/A'):.2f}, {ext.get('lat_max', 'N/A'):.2f}]\n"
        f"Lon: [{ext.get('lon_min', 'N/A'):.2f}, {ext.get('lon_max', 'N/A'):.2f}]\n"
        f"Loss True ratio: {t0_loss.sum()/t0_loss.size:.3f}\n"
        f"ΔSM_surf range: [{t0_tgt[0].min():.4f}, {t0_tgt[0].max():.4f}]\n"
        f"ΔSM_root range: [{t0_tgt[1].min():.4f}, {t0_tgt[1].max():.4f}]"
    )
    ax.text(0.05, 0.95, stats, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate region crop maps (Nature style)")
    parser.add_argument(
        "--out-dir",
        default="figures/regions",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--latlon-nc",
        default="artifacts/geolocation/US_latlon.nc",
        help="Path to US_latlon.nc",
    )
    parser.add_argument(
        "--region-crops-dir",
        default="artifacts/region_crops/US/pt",
        help="Path to region crops directory",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI (default: 300 for publication)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2015,
        help="Year to visualize for detail plots (default: 2015)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib_style()

    # Build region directories mapping
    region_dirs = {}
    for rid in range(1, 7):
        region_id = f"US-R{rid}"
        region_dir = Path(args.region_crops_dir) / region_id
        if region_dir.exists():
            region_dirs[region_id] = region_dir

    print(f"Found {len(region_dirs)} regions: {list(region_dirs.keys())}")

    # ============================================================
    # 1. Generate US overview map with all 6 regions
    # ============================================================
    print("\n[1/2] Generating US region overview map...")
    fig_overview, ax_overview = plt.subplots(
        figsize=(12, 8),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    plot_us_region_overview_map(ax_overview, region_dirs, args.latlon_nc)

    # Add legend for regions
    legend_patches = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=REGION_COLORS[rid],
                    markersize=10, label=f"R{rid}: {REGION_FULL_NAMES[rid]}")
        for rid in range(1, 7)
    ]
    ax_overview.legend(
        handles=legend_patches,
        loc='lower right',
        fontsize=7,
        frameon=True,
        fancybox=False,
        edgecolor=NATURE_COLORS["gray_light"],
        ncol=2,
    )

    out_path_overview = out_dir / "fig_region_map_US.png"
    plt.savefig(out_path_overview, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {out_path_overview}")

    # ============================================================
    # 2. Generate individual region detail figures
    # ============================================================
    print(f"\n[2/2] Generating region detail figures (year={args.year})...")

    for region_id, region_dir in region_dirs.items():
        try:
            fig = plot_region_detail(region_id, region_dir, year=args.year)
            out_path = out_dir / f"fig_{region_id}_detail.png"
            plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
            plt.close()
            print(f"  Saved {out_path}")
        except Exception as e:
            print(f"  ERROR {region_id}: {e}")

    print(f"\nDone. Output directory: {out_dir}")
    print(f"Files: {sorted([p.name for p in out_dir.iterdir()])}")


if __name__ == "__main__":
    main()
