#!/usr/bin/env python3
"""run_one_batch_trial.py -- single isolated subprocess: load the model once,
build a batch of `--batch-size` real LongBench-Pro sequences truncated to
`--length` tokens (cycling through the 50 prepared samples), run ONE naive or
chunked prefill, and print a single JSON result line.

Deliberately a separate process per trial, invoked by run_batch_sweep.py,
because naive prefill at large lengths has been observed to hard-abort via a
Metal-level assertion ("Failed to allocate private MTLBuffer...") rather than
raising a catchable Python RuntimeError -- an uncatchable process abort. A
subprocess-per-trial architecture means a hard abort only kills that one
trial's process; the controller sees a nonzero/signal exit code and records
it as a hard failure without losing the rest of the sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import transformers
from transformers.cache_utils import DynamicCache

MODEL_PATH = os.environ.get("NANBEIGE_MODEL_PATH", "johnhalloran/Nanbeige4.2-3B-mps-fix")
SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "longbench_samples.json")
CHUNK_SIZE = 256


def _naive_prefill(model, input_ids) -> float:
    torch.mps.empty_cache()
    start = time.time()
    with torch.no_grad():
        model(input_ids=input_ids, past_key_values=DynamicCache(), use_cache=True)
    torch.mps.synchronize()
    return time.time() - start


def _chunked_prefill(model, input_ids, chunk_size=CHUNK_SIZE) -> float:
    torch.mps.empty_cache()
    start = time.time()
    total_len = input_ids.shape[1]
    cache = DynamicCache()
    n_full_chunks = (total_len - 1) // chunk_size
    with torch.no_grad():
        for i in range(n_full_chunks):
            s, e = i * chunk_size, i * chunk_size + chunk_size
            cache_position = torch.arange(s, e, device=input_ids.device)
            outputs = model(
                input_ids=input_ids[:, s:e], past_key_values=cache,
                use_cache=True, cache_position=cache_position,
            )
            cache = outputs.past_key_values
        remainder = input_ids[:, n_full_chunks * chunk_size:]
        if remainder.shape[1] > 0:
            cache_position = torch.arange(n_full_chunks * chunk_size, total_len, device=input_ids.device)
            model(input_ids=remainder, past_key_values=cache, use_cache=True, cache_position=cache_position)
    torch.mps.synchronize()
    return time.time() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--method", choices=["naive", "chunked"], required=True)
    parser.add_argument("--sample-offset", type=int, default=0)
    args = parser.parse_args()

    with open(SAMPLES_PATH) as f:
        samples = json.load(f)
    n = len(samples)
    rows = []
    for i in range(args.batch_size):
        s = samples[(args.sample_offset + i) % n]["input_ids"]
        rows.append(s[:args.length])

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="mps", trust_remote_code=True,
    )
    model.eval()

    input_ids = torch.tensor(rows, dtype=torch.long, device=model.device)

    fn = _naive_prefill if args.method == "naive" else _chunked_prefill
    try:
        elapsed = fn(model, input_ids)
        tokens = args.batch_size * args.length
        result = {
            "ok": True, "length": args.length, "batch_size": args.batch_size,
            "method": args.method, "seconds": elapsed, "tokens_per_sec": tokens / elapsed,
        }
    except RuntimeError as e:
        result = {"ok": False, "length": args.length, "batch_size": args.batch_size,
                   "method": args.method, "error": str(e)}

    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
