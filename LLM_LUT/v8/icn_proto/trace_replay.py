#!/usr/bin/env python3
"""Replay one block's lifecycle from a trace dir (icn_proto.trace).

Usage:
    python -m icn_proto.trace_replay <trace_dir> <name-substring>
    python -m icn_proto.trace_replay <trace_dir> --summary

The first form prints every event mentioning the block, merged across
all processes in time order — the block's complete life (publish /
evict and where it landed / drops / recalls / demands / fetch / outcome).
The second form prints per-event-type counts per process, a sanity
check that the instrument saw what the run actually did.

Mechanism verdicts are read off the first form: does the chain go
evict->redemand->recall (tier caught it) or evict->redemand->recompute
(tier missed it), and where did any copy land. No aggregation beyond
listing.

Run:  python -m icn_proto.trace_replay
"""

import glob
import json
import os
import sys
from collections import Counter


def load_events(trace_dir):
    evs = []
    for p in sorted(glob.glob(os.path.join(trace_dir, "*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    evs.append(json.loads(line))
    evs.sort(key=lambda e: (e["t"], e["who"], e["seq"]))
    return evs


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    trace_dir, arg = sys.argv[1], sys.argv[2]
    evs = load_events(trace_dir)
    if not evs:
        sys.exit(f"no events under {trace_dir}")
    if arg == "--summary":
        c = Counter((e["who"], e["event"]) for e in evs)
        for (who, ev), n in sorted(c.items()):
            print(f"  {who:8s} {ev:16s} {n}")
        print(f"total {len(evs)} events, "
              f"{evs[-1]['t'] - evs[0]['t']:.1f}s span")
        return
    hits = [e for e in evs if arg in e.get("name", "")]
    if not hits:
        sys.exit(f"no events for name containing {arg!r}")
    for e in hits:
        d = e.get("detail")
        ds = ("  " + json.dumps(d, ensure_ascii=False)) if d else ""
        print(f"{e['t']:.3f} {e['who']:8s} {e['event']:16s} {ds}")


if __name__ == "__main__":
    main()
