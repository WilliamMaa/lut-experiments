#!/usr/bin/env bash
# Aggressively delete ALL fine-tune checkpoints under a results directory.
# Keeps summary.json and other small files.
# Usage:
#   bash nuke_checkpoints.sh /data/mingyu/LLM_LUT/v5/results
#   bash nuke_checkpoints.sh /data/mingyu/LLM_LUT/v4/results

set -e

ROOT="${1:-/data/mingyu/LLM_LUT/v5/results}"

if [ ! -d "$ROOT" ]; then
    echo "Directory does not exist: $ROOT"
    exit 1
fi

echo "WARNING: This will delete ALL .pt checkpoint files and epoch LUT directories under"
echo "  $ROOT"
echo "Only summary.json and non-checkpoint files will be kept."
echo ""
read -p "Type YES to proceed: " confirm
if [ "$confirm" != "YES" ]; then
    echo "Aborted."
    exit 1
fi

BEFORE=$(du -sb "$ROOT" | awk '{print $1}')

# Delete .pt checkpoint files
echo "Deleting .pt files..."
find "$ROOT" -type f -name '*.pt' -print -delete

# Delete epoch LUT directories (named like l15_epoch1_down_lut, l15_epoch1_o_lut, etc.)
echo "Deleting epoch LUT directories..."
find "$ROOT" -type d -name '*_epoch*_lut' -print -exec rm -rf {} +

# Delete any remaining per-epoch checkpoint dirs (just in case)
echo "Deleting per-epoch checkpoint directories..."
find "$ROOT" -type d \( -name '*_epoch*_down_proj' -o -name '*_epoch*_o_proj' \) -print -exec rm -rf {} +

AFTER=$(du -sb "$ROOT" | awk '{print $1}')
FREED=$((BEFORE - AFTER))

echo ""
echo "Done."
echo "Before: $((BEFORE / 1024**3)) GiB"
echo "After:  $((AFTER / 1024**3)) GiB"
echo "Freed:  $((FREED / 1024**3)) GiB"
