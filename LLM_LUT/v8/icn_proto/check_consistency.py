#!/usr/bin/env python3
"""E1 consistency check for the block-chain protocol (05 v3 §6 step 1).

Turns whose prefixes have the SAME content fingerprint (fp = chained block
hash at floor(cum)) must produce IDENTICAL decoded_ids no matter where or
how they ran: fresh full prefill, local resume, or cross-worker fetch +
inject. Any disagreement means the block transfer / checkpoint / inject
path corrupted model state.

    python -m icn_proto.check_consistency [results/icn_proto/blkcluster_*.json]

Exit code 0 iff every fp group with >1 completed record fully agrees.
"""

import glob
import json
import os
import sys


def check(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    recs = [r for r in d.get("records", [])
            if r.get("ok") and r.get("decoded_ids")]
    groups = {}
    for r in recs:
        groups.setdefault(r.get("fp"), []).append(r)

    print(f"file: {path}")
    print(f"  completed turns with decode: {len(recs)}, "
          f"fp groups: {len(groups)}")
    n_bad = 0
    for fp, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(rs) < 2:
            continue
        ref = rs[0]["decoded_ids"]
        agree = sum(1 for r in rs if r["decoded_ids"] == ref)
        modes = {r.get("decision", {}).get("mode", "?") for r in rs}
        workers = sorted({r["worker"] for r in rs})
        es = [r.get("E") for r in rs]
        status = "AGREE" if agree == len(rs) else "DISAGREE"
        if agree != len(rs):
            n_bad += 1
        print(f"  [{status}] group fp=...{str(fp)[-12:]} n={len(rs)} "
              f"agree={agree}/{len(rs)} modes={sorted(modes)} "
              f"workers={workers} E={es}")
        if agree != len(rs):
            for r in rs:
                print(f"      {r['request_id']:<10} w={r['worker']} "
                      f"E={r.get('E')} decode={r['decoded_ids']}")
    print(f"  => {'CONSISTENT' if n_bad == 0 else f'{n_bad} DISAGREEING groups'}")
    return n_bad


def main():
    paths = sys.argv[1:]
    if not paths:
        base = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "results", "icn_proto")
        paths = sorted(glob.glob(os.path.join(base, "blkcluster_*.json")))[-1:]
    bad = 0
    for p in paths:
        bad += check(p)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
