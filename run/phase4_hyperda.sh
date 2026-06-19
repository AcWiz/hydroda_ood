#!/bin/bash
# compatibility wrapper for the staged HyperDA source-stage mainline.
#
# Active paper-facing source-stage entrypoint:
#   run/phase4_hyperda_staged.sh
#
# Historical scratch HyperDA source-stage training is retired from active run
# entries. This wrapper preserves old command muscle memory while requiring
# the staged protocol arguments:
#
#   bash run/phase4_hyperda.sh auto US-R1 0 0
#   bash run/phase4_hyperda.sh /path/to/source.pt US-R1 0 0

set -euo pipefail

if [[ "${1:-}" =~ ^[A-Z]{2}-R[0-9]+$ ]]; then
    exec bash "$(dirname "$0")/phase4_hyperda_staged.sh" auto "$@"
fi

exec bash "$(dirname "$0")/phase4_hyperda_staged.sh" "$@"
