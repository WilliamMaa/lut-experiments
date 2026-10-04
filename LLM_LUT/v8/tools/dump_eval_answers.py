"""Dump sample Q/GT/answer triples from an eval_longctx_server.py result JSON.

Purpose (docs/28 §8 step 3): after a 对拍 run, distinguish
  - coherent-but-wrong answers  -> method collapse (retention too small),
                                   respond with a slots sweep
  - garbage/repeated answers    -> real bug in the compressed path,
                                   open the plugin debugger
without eyeballing the raw JSON by hand.

Usage (remote, from LLM_LUT/v8):
    python tools/dump_eval_answers.py --results results/eval_64k_compressed.json
    python tools/dump_eval_answers.py --results results/eval_64k_compressed.json --n 12 --full
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="result JSON written by eval_longctx_server.py")
    ap.add_argument("--n", type=int, default=6,
                    help="how many question samples to print")
    ap.add_argument("--full", action="store_true",
                    help="print full answers (default: first 200 chars)")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        d = json.load(f)

    print(f"results={args.results}")
    print(f"fact_acc={d.get('fact_acc'):.4f} "
          f"({d.get('n_correct')}/{d.get('n_total')})")
    per_qtype = d.get("per_qtype", {})
    for qt, v in sorted(per_qtype.items()):
        print(f"  {qt:>18s}: {v['correct']}/{v['total']}")
    print("=" * 60)

    rows = [r for r in d.get("results", []) if "question" in r]
    if not rows:
        print("no per-question rows (anchor failure?)")
        for r in d.get("results", [])[:args.n]:
            print(r)
        return

    shown = 0
    for r in rows:
        if shown >= args.n:
            break
        ans = str(r.get("answer", ""))
        if not args.full:
            ans = ans[:200].replace("\n", " ")
        print(f"[doc {r['doc_index']} q{r['q_index']}] "
              f"{r['qtype']} correct={r['correct']} {r.get('seconds', 0)}s")
        print(f"  Q : {r['question'][:80]}")
        print(f"  GT: {r['gt']}")
        print(f"  A : {ans}")
        print("-" * 60)
        shown += 1


if __name__ == "__main__":
    main()
