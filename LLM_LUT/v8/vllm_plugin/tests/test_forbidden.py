"""Forbidden-pattern guard (docs/32 禁止项): scans the plugin source so
the banned failure modes cannot silently return. Run:
python vllm_plugin/tests/test_forbidden.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

CORE = ["vllm_plugin/impl.py", "vllm_plugin/backend.py",
        "vllm_plugin/blockplan.py"]

# (regex, files, why) — a match anywhere is a FAIL.
BANNED = [
    # capacity clamps: required must never be silently cut to available
    (r"min\s*\(\s*required", CORE, "capacity clamp min(required, ...)"),
    (r"min\s*\(\s*available", CORE, "capacity clamp min(available, ...)"),
    # request-identity heuristics (docs/31 I4 ban list)
    (r"blocks\s*\[\s*0\s*\]", ["vllm_plugin/impl.py",
                               "vllm_plugin/backend.py"],
     "identity heuristic blocks[0]"),
    (r"seq_lens", ["vllm_plugin/impl.py", "vllm_plugin/backend.py"],
     "lag-prone seq_lens field"),
    (r"num_computed\s*==\s*0", ["vllm_plugin/impl.py",
                                "vllm_plugin/backend.py"],
     "'computed==0 means new request' heuristic"),
]

# Presence requirements: the docs/32 single-plan architecture must exist.
REQUIRED = [
    (r"plan_write_span\(", ["vllm_plugin/backend.py"],
     "builder must produce the per-request plan"),
    (r"md\.block_plans\s*=", ["vllm_plugin/backend.py"],
     "builder must attach block_plans to metadata"),
    (r"certify_kernel\(", ["vllm_plugin/impl.py"],
     "impl must consume the plan via certify_kernel"),
]

# The impl must never build its own plan (docs/32 §2).
IMPL_BANS = [
    (r"build_block_plan\(", "impl re-derives the plan instead of consuming"),
]


def scan():
    fails = []
    for pat, files, why in BANNED:
        for rel in files:
            path = os.path.join(ROOT, rel)
            src = open(path).read()
            for i, line in enumerate(src.splitlines(), 1):
                if re.search(pat, line):
                    fails.append(f"{rel}:{i}: {why}: {line.strip()}")
    for pat, files, why in REQUIRED:
        for rel in files:
            src = open(os.path.join(ROOT, rel)).read()
            if not re.search(pat, src):
                fails.append(f"{rel}: MISSING {why} ({pat})")
    impl_src = open(os.path.join(ROOT, "vllm_plugin/impl.py")).read()
    for pat, why in IMPL_BANS:
        m = re.search(pat, impl_src)
        if m:
            line = impl_src[:m.start()].count("\n") + 1
            fails.append(f"vllm_plugin/impl.py:{line}: {why}")
    return fails


def main():
    fails = scan()
    if fails:
        print("[forbidden] FAIL")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("[forbidden] ALL PASS")


if __name__ == "__main__":
    main()
