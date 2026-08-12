#!/usr/bin/env python3
"""run_batch_sweep.py -- controller for the naive-vs-chunked batched
throughput experiment across lengths [1024, 2048, 4096, 8192, 9205, 10218,
11231, 12244], on 50 real LongBench-Pro documents (no concatenation -- see
prepare_longbench_samples.py). The last four lengths were added after the
first pass to pinpoint chunked prefill's real single-request ceiling between
the known-good 8192 and the known-failing M=12244 (the exact token length of
the reproduced production OOM in reproduce_original_oom.py), rather than
leaving that span uncharacterized.

For each (length, method), doubles batch size (1, 2, 4, ...) via an isolated
subprocess per attempt (run_one_batch_trial.py) until a trial fails --
whether a clean "ok": false result (caught RuntimeError OOM) or a hard
process abort (nonzero/signal exit, e.g. Metal's uncatchable
"Failed to allocate private MTLBuffer" assertion) -- then re-runs the last
successful (length, method, batch_size) 3 times to report a stable mean/std
throughput at each method's own real max batch size.

This is intentionally NOT imported as a library and run in-process: every
single trial is its own subprocess specifically so a hard Metal abort in one
trial cannot take down the rest of the sweep.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

LENGTHS = [1024, 2048, 4096, 8192, 9205, 10218, 11231, 12244]
METHODS = ["naive", "chunked"]
MAX_BATCH_SIZE = 64
N_REPEATS = 3
WORKER = os.path.join(os.path.dirname(__file__), "run_one_batch_trial.py")
OUT_PATH = os.path.join(os.path.dirname(__file__), "batch_sweep_results.json")


def _run_trial(length, batch_size, method, sample_offset=0, timeout=900):
    cmd = [
        sys.executable, "-u", WORKER,
        "--length", str(length), "--batch-size", str(batch_size),
        "--method", method, "--sample-offset", str(sample_offset),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "hard_failure": True, "reason": "timeout"}
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        return {"ok": False, "hard_failure": True, "reason": f"exit={proc.returncode}: {tail}"}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    return {"ok": False, "hard_failure": True, "reason": "no RESULT_JSON in output"}


def _find_max_batch(length, method):
    batch_size = 1
    last_good = None
    while batch_size <= MAX_BATCH_SIZE:
        print(f"[{method} len={length}] probing batch={batch_size} ...", flush=True)
        result = _run_trial(length, batch_size, method)
        if result.get("ok"):
            print(f"  -> ok: {result['seconds']:.2f}s, {result['tokens_per_sec']:.1f} tok/s", flush=True)
            last_good = batch_size
            batch_size *= 2
        else:
            reason = result.get("error") or result.get("reason")
            print(f"  -> FAILED: {reason}", flush=True)
            return last_good, batch_size, reason
    return last_good, None, None


def main():
    all_results = {}
    for length in LENGTHS:
        all_results[length] = {}
        for method in METHODS:
            max_batch, fail_batch, fail_reason = _find_max_batch(length, method)
            entry = {"max_batch": max_batch, "first_fail_batch": fail_batch, "fail_reason": fail_reason}
            if max_batch is not None:
                trials = []
                for r in range(N_REPEATS):
                    res = _run_trial(length, max_batch, method, sample_offset=r * max_batch)
                    if res.get("ok"):
                        trials.append(res)
                        print(f"[{method} len={length}] repeat {r}: {res['seconds']:.2f}s, "
                              f"{res['tokens_per_sec']:.1f} tok/s", flush=True)
                    else:
                        print(f"[{method} len={length}] repeat {r}: FAILED on repeat "
                              f"({res.get('error') or res.get('reason')})", flush=True)
                entry["repeats"] = trials
                if trials:
                    times = [t["seconds"] for t in trials]
                    tps = [t["tokens_per_sec"] for t in trials]
                    entry["seconds_mean"] = sum(times) / len(times)
                    entry["tokens_per_sec_mean"] = sum(tps) / len(tps)
            all_results[length][method] = entry
            with open(OUT_PATH, "w") as f:
                json.dump(all_results, f, indent=2)
    print(f"\nwrote {OUT_PATH}")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
