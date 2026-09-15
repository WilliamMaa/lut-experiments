"""Result analysis tool for v8 KV-cache / VQK runs.

Usage:
  python tools/analyze_result.py results/xxx.json
      Summary of one run: config check, storage, metrics, decode, sentinel turns.

  python tools/analyze_result.py results/A.json --compare results/B.json
      Turn-level comparison (determinism check / A-B diff): aggregate diffs,
      EOS flips, per-turn text divergence points.

Config conventions enforced by the check block:
  - k_bits/v_bits in storage_stats must match the filename token (k8v8 / k4v4),
    else [WARN] is printed (there has been one filename/config mismatch before).
  - compression_ratio is read from storage_stats.
Sentinel turns (v3 multi-turn set) are (0,4), (0,5), (3,4).
"""
import argparse
import json
import os
import sys

SENTINEL_TURNS = [(0, 4), (0, 5), (3, 4)]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def flat_turns(data, side):
    """{(doc_idx, turn): turn_dict} for baseline or patched."""
    out = {}
    for di, doc in enumerate(data[side].get("generation", [])):
        for t in doc.get("turns", []):
            out[(di, t["turn"])] = t
    return out


def fmt_metrics(m):
    return ("eos %.4f  rep %.4f  len %.2f" % (
        m.get("eos_success_rate", 0.0),
        m.get("repetition_rate", 0.0),
        m.get("avg_output_length", 0.0)))


def summarize(path, data):
    print("=" * 60)
    print("file:", os.path.basename(path))
    print("patch:", data.get("patch_name") or data.get("patch"))
    cfg = data.get("config", {})
    if cfg:
        print("config:", json.dumps(cfg, ensure_ascii=False))

    s = data.get("storage_stats", {})
    print("storage: k_bits=%s v_bits=%s ratio=%s" % (
        s.get("k_bits"), s.get("v_bits"), s.get("compression_ratio")))
    # filename/config consistency check
    base = os.path.basename(path)
    for tok, kb, vb in (("k8v8", 8, 8), ("k4v4", 4, 4), ("k16v16", 16, 16)):
        if tok in base:
            if s.get("k_bits") != kb or s.get("v_bits") != vb:
                print("[WARN] filename says %s but storage says k%s v%s"
                      % (tok, s.get("k_bits"), s.get("v_bits")))
    print("PPL: baseline %.6f  patched %.6f  delta %.6f" % (
        data.get("baseline", {}).get("ppl", 0.0),
        data.get("patched", {}).get("ppl", 0.0),
        data.get("delta", {}).get("ppl", 0.0)))
    print("baseline:", fmt_metrics(data.get("baseline", {}).get("generation_metrics", {})))
    print("patched :", fmt_metrics(data.get("patched", {}).get("generation_metrics", {})))
    dm = data.get("decode_metrics", {})
    if dm:
        print("decode: KL %.4f  top1 %.4f  top5 %.4f  teacher-prob %.4f  (n=%s)" % (
            dm.get("avg_decode_kl", 0.0), dm.get("decode_top1_agreement", 0.0),
            dm.get("decode_top5_agreement", 0.0),
            dm.get("avg_teacher_greedy_token_prob_under_student", 0.0),
            dm.get("total_decode_positions")))

    pt = flat_turns(data, "patched")
    if pt:
        print("--- sentinel turns (patched) ---")
        for k in SENTINEL_TURNS:
            t = pt.get(k)
            if t:
                print("doc%d T%d [eos=%s]: %s" % (
                    k[0], k[1], t.get("ended_with_eos"),
                    repr(t.get("output", "")[:100])))
            else:
                print("doc%d T%d: MISSING" % k)


def compare(path_a, data_a, path_b, data_b):
    print("=" * 60)
    print("compare:", os.path.basename(path_a), "vs", os.path.basename(path_b))
    same_agg = True
    for side in ("baseline", "patched"):
        ma = data_a.get(side, {}).get("generation_metrics", {})
        mb = data_b.get(side, {}).get("generation_metrics", {})
        for key in ("eos_success_rate", "repetition_rate", "avg_output_length"):
            if ma.get(key) != mb.get(key):
                same_agg = False
                print("AGG-DIFF %s.%s: %s vs %s" % (side, key, ma.get(key), mb.get(key)))
    ka = data_a.get("decode_metrics", {}).get("avg_decode_kl")
    kb = data_b.get("decode_metrics", {}).get("avg_decode_kl")
    if ka != kb:
        same_agg = False
        print("AGG-DIFF decode KL: %s vs %s" % (ka, kb))
    if same_agg:
        print("aggregate metrics: IDENTICAL")

    n_diff = 0
    for side in ("baseline", "patched"):
        fa, fb = flat_turns(data_a, side), flat_turns(data_b, side)
        for k in sorted(set(fa) | set(fb)):
            ta, tb = fa.get(k), fb.get(k)
            if ta is None or tb is None:
                print("TURN-MISSING", side, k)
                n_diff += 1
                continue
            if ta.get("output") != tb.get("output") or \
               ta.get("ended_with_eos") != tb.get("ended_with_eos"):
                n_diff += 1
                oa, ob = ta.get("output", ""), tb.get("output", "")
                i = 0
                while i < min(len(oa), len(ob)) and oa[i] == ob[i]:
                    i += 1
                print("DIFF %s doc%d T%d  eos %s->%s  common-prefix %d/%d/%d chars" % (
                    side, k[0], k[1], ta.get("ended_with_eos"),
                    tb.get("ended_with_eos"), i, len(oa), len(ob)))
                print("  A: ...%s" % repr(oa[max(0, i - 20):i + 60]))
                print("  B: ...%s" % repr(ob[max(0, i - 20):i + 60]))
    print("turn diffs: %d / %d turns" % (n_diff, len(flat_turns(data_a, "patched"))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="result JSON to summarize")
    ap.add_argument("--compare", metavar="B.json",
                    help="second result JSON for turn-level comparison")
    args = ap.parse_args()

    data_a = load(args.file)
    if args.compare:
        compare(args.file, data_a, args.compare, load(args.compare))
    else:
        summarize(args.file, data_a)


if __name__ == "__main__":
    sys.exit(main())
