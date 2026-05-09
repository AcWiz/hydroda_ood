"""Generate US Region Masks Preview Visualization."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def main():
    parser = argparse.ArgumentParser(description="Generate US region masks preview")
    parser.add_argument(
        "--masks-nc",
        default="artifacts/regions/US_region_masks.nc",
        help="Path to US_region_masks.nc",
    )
    parser.add_argument(
        "--out-png",
        default="figures/regions/US_region_masks_preview.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.masks_nc)
    mask_int = ds["region_mask_integer"].values
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    region_names = {
        1: "US-R1: Southwest Desert",
        2: "US-R2: S. Great Plains",
        3: "US-R3: CA Central Valley",
        4: "US-R4: Corn Belt",
        5: "US-R5: Southeast US",
        6: "US-R6: Central Rockies",
    }
    colors = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx in range(1, 7):
        ax = axes[idx - 1]
        region_mask = mask_int == idx

        # Plot base map (all regions colored)
        im = ax.pcolormesh(lon, lat, mask_int, cmap="Set1", vmin=0, vmax=6)
        ax.pcolormesh(lon, lat, region_mask, cmap="Greens", alpha=0.7)
        ax.set_title(region_names[idx])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(lon.min(), lon.max())
        ax.set_ylim(lat.min(), lat.max())

    # Overlay plot showing all regions
    ax_overlay = fig.add_subplot(3, 1, 3)
    im = ax_overlay.pcolormesh(lon, lat, mask_int, cmap="Set1", vmin=0, vmax=6)
    ax_overlay.set_title("All US Regions (R1-R6)")
    ax_overlay.set_xlabel("Longitude")
    ax_overlay.set_ylabel("Latitude")
    cbar = plt.colorbar(im, ax=ax_overlay, ticks=range(7))
    cbar.set_label("Region ID")

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150, bbox_inches="tight")
    plt.close()
    ds.close()
    print(f"Saved: {args.out_png}")


if __name__ == "__main__":
    main()