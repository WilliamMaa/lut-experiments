#!/usr/bin/env python3
"""Block lifecycle tracing: a per-process append-only JSONL event stream.

Purpose (14-icn-mechanism-matrix): every mechanism in this project is a
causal chain over blocks —

    produced -> residency (copy? evict? where to?)
    -> later demand -> (local hit / cross-worker fetch / tier recall /
       recompute)

and mechanism verdicts come from REPLAYING those chains, not from
aggregate counters. This module is the instrument: each process emits
structured events; trace_replay.py merges the streams and prints one
block's complete life.

Events are emitted only when --trace-dir is set; Tracer is a no-op
otherwise, so production runs pay nothing. One file per process
(sched.jsonl, w0.jsonl, ...) inside the given directory — no protocol
change, no cross-process ordering requirement (replay merges by
timestamp with per-process sequence tiebreak).
"""

import json
import os
import time


def open_tracer(who, path=None):
    """Returns a Tracer for `who`. path defaults to
    $ICN_TRACE_DIR/<who>.jsonl; disabled (writes nowhere) when tracing
    is off or the dir is unavailable."""
    if path is None:
        d = os.environ.get("ICN_TRACE_DIR")
        if not d:
            return Tracer(None, who)
        path = os.path.join(d, f"{who}.jsonl")
    return Tracer(path, who)


class Tracer:
    def __init__(self, path, who):
        self.who = who
        self._seq = 0
        self._f = None
        if path:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self._f = open(path, "a", encoding="utf-8",
                               buffering=1)
            except OSError:
                # tracing must never crash an experiment run
                self._f = None

    @property
    def enabled(self):
        return self._f is not None

    def emit(self, event, name=None, **detail):
        if self._f is None:
            return
        self._seq += 1
        rec = {"seq": self._seq, "t": time.time(), "who": self.who,
               "event": event}
        if name is not None:
            rec["name"] = name
        if detail:
            rec["detail"] = detail
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None
