#!/usr/bin/env python3
"""Visualize US region crops for acceptance verification.
Generates PNG reports for each region showing mask and sample data.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import os
from pathlib import Path

ARTIFACTS = Path("artifacts/region_crops/US")
OUT_DIR = Path("artifacts/region_crops/US/visualization")
OUT_DIR.mkdir(exist_ok=True, parents=True)

REGIONS = [f"US-R{i}" for i in range(1, 7)]
YEARS = list(range(2015, 2026))

def load_json(path):
    with open(path) as f:
        return json.load(f)

def plot_region_overview(region_id: str, axs, region_dir: Path, year=2015):
    """Plot region mask + data overview."""
    # Load mask
    mask = torch.load(region_dir / "region_mask.pt").numpy()
    H, W = mask.shape

    # Load a sample input snapshot (t=0)
    inp = torch.load(region_dir / "da_full" / str(year) / "input.pt")
    # inp shape: [T, 12, H, W] -> pick first time step
    t0 = inp[0].numpy()  # [12, H, W]

    # Load target increment snapshot (t=0)
    tgt = torch.load(region_dir / "da_full" / str(year) / "target_increment.pt")
    t0_tgt = tgt[0].numpy()  # [2, H, W]

    # Load loss mask
    loss = torch.load(region_dir / "da_full" / str(year) / "loss_mask.pt")
    t0_loss = loss[0].numpy()  # [H, W]

    # --- Top row: mask + loss mask ---
    ax = axs[0, 0]
    ax.imshow(mask, cmap='tab20', vmin=0, vmax=20)
    ax.set_title(f"{region_id} region_mask.pt\nshape={H}x{W}", fontsize=8)
    ax.axis('off')

    ax = axs[0, 1]
    ax.imshow(t0_loss.astype(float), cmap='gray')
    ax.set_title("loss_mask (t=0)", fontsize=8)
    ax.axis('off')

    # --- Data channels (input channels 0,1,2) ---
    for c, ch_name in enumerate(['SM_surface', 'SM_rootzone', 'soil_temp']):
        ax = axs[1, c]
        im = ax.imshow(t0[c], cmap='viridis')
        ax.set_title(f"input ch{c} ({ch_name}) t=0", fontsize=8)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.03)

    # --- Target increments ---
    for c, ch_name in enumerate(['ΔSM_surface', 'ΔSM_rootzone']):
        ax = axs[2, c]
        vmax = np.abs(t0_tgt[c]).max()
        im = ax.imshow(t0_tgt[c], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f"target_increment ch{c} ({ch_name}) t=0", fontsize=8)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.03)

    # --- Summary stats ---
    ax = axs[2, 2]
    ax.axis('off')
    stats = (
        f"Temporal samples (2015): {inp.shape[0]}\n"
        f"Input shape: {inp.shape} (T,C,H,W)\n"
        f"Target increment shape: {tgt.shape}\n"
        f"loss_mask True ratio: {t0_loss.sum()/t0_loss.size:.3f}\n"
        f"Mask unique values: {np.unique(mask)}"
    )
    ax.text(0.05, 0.95, stats, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace')


def main():
    print("Generating visualization PNGs for 6 US regions...")

    for region_id in REGIONS:
        region_dir = ARTIFACTS / "pt" / region_id
        if not region_dir.exists():
            print(f"  SKIP {region_id}: not found at {region_dir}")
            continue

        # Load metadata
        meta_path = region_dir / "metadata.json"
        meta = load_json(meta_path) if meta_path.exists else {}

        fig, axs = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f"{region_id} — {meta.get('region_name', region_id)}", fontsize=12, fontweight='bold')

        try:
            plot_region_overview(region_id, axs, region_dir, year=2015)
        except Exception as e:
            print(f"  ERROR {region_id}: {e}")
            fig.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            for ax in axs.flat:
                ax.axis('off')

        # Add region info
        region_info = meta.get('resolved_extent', {})
        fig.text(0.99, 0.01,
                 f"lat: {region_info.get('lat', 'N/A')}\nlon: {region_info.get('lon', 'N/A')}",
                 ha='right', va='bottom', fontsize=7, color='gray')

        out_path = OUT_DIR / f"{region_id}_overview.png"
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out_path}")

    # Generate comparison grid
    print("Generating comparison grid...")
    fig, axs = plt.subplots(2, 6, figsize=(24, 8))
    for col, region_id in enumerate(REGIONS):
        region_dir = ARTIFACTS / "pt" / region_id
        mask_path = region_dir / "region_mask.pt"
        if not mask_path.exists():
            continue
        mask = torch.load(mask_path).numpy()

        ax = axs[0, col]
        ax.imshow(mask, cmap='tab20', vmin=0, vmax=20)
        ax.set_title(f"{region_id}\n{mask.shape}", fontsize=9)
        ax.axis('off')

        # load loss mask ratio heatmap for 2015
        try:
            loss = torch.load(region_dir / "da_full" / "2015" / "loss_mask.pt")
            ratio = loss.float().mean(dim=0).numpy()
            ax2 = axs[1, col]
            im = ax2.imshow(ratio, cmap='hot', vmin=0, vmax=1)
            ax2.set_title(f"loss True ratio", fontsize=8)
            ax2.axis('off')
        except Exception:
            pass

    fig.suptitle("US Regions — Mask & Loss Coverage (2015)", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    grid_path = OUT_DIR / "us_regions_comparison.png"
    plt.savefig(grid_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {grid_path}")

    print(f"\nDone. Output: {OUT_DIR}")
    print("Files:", sorted(os.listdir(OUT_DIR)))


if __name__ == "__main__":
    main()