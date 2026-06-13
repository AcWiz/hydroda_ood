#!/usr/bin/env bash
# Hold CUDA memory in the foreground. Ctrl-C releases it.

set -euo pipefail

GPU=0
GB=12
CHUNK_MIB=256
MIN_FREE_GB=2
HEARTBEAT_SEC=30
MAX_HOURS=""
FORCE=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/utils/gpu_memory_hold.sh [options]

Hold CUDA memory in the foreground. Press Ctrl-C to exit and release memory.

Options:
  --gpu ID              GPU id to use. Default: 0
  --gb GB               GPU memory to hold. Default: --gb 12
  --chunk-mib MIB       Allocation chunk size. Default: 256
  --min-free-gb GB      Minimum free memory left after allocation. Default: 2
  --heartbeat-sec SEC   Status print interval. Default: 30
  --max-hours HOURS     Exit automatically after this many hours.
  --force               Skip the free-memory safety check.
  -h, --help            Show this help.

Examples:
  bash scripts/utils/gpu_memory_hold.sh --gpu 1
  bash scripts/utils/gpu_memory_hold.sh --gpu 1 --gb 11
  bash scripts/utils/gpu_memory_hold.sh --gpu 1 --gb 12
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="${2:?missing --gpu value}"; shift 2 ;;
        --gb) GB="${2:?missing --gb value}"; shift 2 ;;
        --chunk-mib) CHUNK_MIB="${2:?missing --chunk-mib value}"; shift 2 ;;
        --min-free-gb) MIN_FREE_GB="${2:?missing --min-free-gb value}"; shift 2 ;;
        --heartbeat-sec) HEARTBEAT_SEC="${2:?missing --heartbeat-sec value}"; shift 2 ;;
        --max-hours) MAX_HOURS="${2:?missing --max-hours value}"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

export HOLD_GB="$GB" HOLD_CHUNK_MIB="$CHUNK_MIB" HOLD_MIN_FREE_GB="$MIN_FREE_GB"
export HOLD_HEARTBEAT_SEC="$HEARTBEAT_SEC" HOLD_MAX_HOURS="$MAX_HOURS" HOLD_FORCE="$FORCE"

CUDA_VISIBLE_DEVICES="$GPU" python - "$GPU" <<'PY'
import os, signal, socket, sys, time
import torch

gpu = sys.argv[1]
gb = float(os.environ["HOLD_GB"])
chunk_mib = int(os.environ["HOLD_CHUNK_MIB"])
min_free_gb = float(os.environ["HOLD_MIN_FREE_GB"])
heartbeat_sec = float(os.environ["HOLD_HEARTBEAT_SEC"])
max_hours = os.environ["HOLD_MAX_HOURS"]
force = os.environ["HOLD_FORCE"] == "1"
stop = False
GiB = 1024 ** 3
MiB = 1024 ** 2

def on_signal(signum, _frame):
    global stop
    stop = True
    print(f"\nReceived {signal.Signals(signum).name}; releasing memory...", flush=True)

signal.signal(signal.SIGINT, on_signal)
signal.signal(signal.SIGTERM, on_signal)

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available to PyTorch")

torch.cuda.set_device(0)
free, total = torch.cuda.mem_get_info()
target = int(gb * GiB)
if not force and target > free - int(min_free_gb * GiB):
    raise SystemExit(
        "ERROR: requested hold is too large: "
        f"requested={gb:.2f} GB free={free/GiB:.2f} GB min_free={min_free_gb:.2f} GB"
    )

print(
    f"holding GPU memory: pid={os.getpid()} host={socket.gethostname()} gpu={gpu} "
    f"target={gb:.2f} GB free_before={free/GiB:.2f}/{total/GiB:.2f} GB",
    flush=True,
)

chunks, held = [], 0
start = next_beat = time.monotonic()
deadline = start + float(max_hours) * 3600 if max_hours else None

try:
    while held < target and not stop:
        n = min(chunk_mib * MiB, target - held)
        chunks.append(torch.empty(n, dtype=torch.uint8, device="cuda"))
        held += n
    torch.cuda.synchronize()

    free, _ = torch.cuda.mem_get_info()
    print(f"active: gpu={gpu} held={held/GiB:.2f} GB free_after={free/GiB:.2f} GB. Ctrl-C to release.", flush=True)

    while not stop and (deadline is None or time.monotonic() < deadline):
        now = time.monotonic()
        if now >= next_beat:
            free, _ = torch.cuda.mem_get_info()
            print(f"heartbeat: gpu={gpu} held={held/GiB:.2f} GB free={free/GiB:.2f} GB", flush=True)
            next_beat = now + heartbeat_sec
        time.sleep(0.5)
finally:
    chunks.clear()
    torch.cuda.empty_cache()
    print("released GPU memory.", flush=True)
PY
