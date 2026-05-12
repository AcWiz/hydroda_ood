#!/bin/bash
# Create HydroDA-OOD conda environments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=== Creating HydroDA-OOD GPU environment (hydroda-ood) ==="
conda env create -f environment-gpu.yml

echo ""
echo "=== Creating HydroDA-OOD CPU fallback environment (hydroda-ood-cpu) ==="
conda env create -f environment.yml -n hydroda-ood-cpu

echo ""
echo "=== Done ==="
echo "GPU environment: conda activate hydroda-ood"
echo "CPU environment: conda activate hydroda-ood-cpu"
