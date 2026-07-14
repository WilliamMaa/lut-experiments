#!/bin/bash
# clear_sensitivity_scan.sh
# Removes sensitivity scan output so you can restart from scratch.
# Use this when you want to change scan params and re-run.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$SCRIPT_DIR/results/sensitivity_scan.json"

if [ -f "$OUT" ]; then
    echo "Removing $OUT ..."
    rm -f "$OUT"
    echo "Done. You can now re-run scan_module_sensitivity.py with new settings."
else
    echo "$OUT does not exist. Nothing to clear."
fi
