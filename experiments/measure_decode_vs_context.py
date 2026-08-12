#!/usr/bin/env python3
"""measure_decode_vs_context.py -- controlled measurement of decode
throughput (tokens/sec generated) as a function of existing context length,
using real LongBench-Pro text as the prefix (via chunked prefill, matching
the harness), then forcing a fixed number of new tokens with
min_new_tokens=max_new_tokens so early stopping cannot bias the measurement.

Motivation: an end-to-end MCPMark replay showed decode collapsing to 1.89
tok/s at ~5100 tokens of context, versus ~12-16 tok/s measured earlier at a
trivial (~50 token) prompt. This isolates whether that slowdown is a
genuine, reproducible function of context length (consistent with the
Looped Transformer's doubled per-token compute, Section 3) rather than a
one-off artifact of that specific replay.
"""
from __future__ import annotations

import json
import os
import time

import torch
import transformers
from transformers.cache_utils import DynamicCache

MODEL_PATH = os.environ.get("NANBEIGE_MODEL_PATH", "johnhalloran/Nanbeige4.2-3B-mps-fix")
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "longbench_samples.json")
CHUNK_SIZE = 256
CONTEXT_LENGTHS = [64, 1024, 5120, 8192]
NEW_TOKENS = 64


def _chunked_prefill(model, input_ids, chunk_size=CHUNK_SIZE):
    cache = DynamicCache()
    total_len = input_ids.shape[1]
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
    remainder_start = n_full_chunks * chunk_size
    return cache, remainder_start


def main():
    with open(CORPUS_PATH) as f:
        samples = json.load(f)
    long_ids = samples[0]["input_ids"]

    print(f"loading {MODEL_PATH} ...", flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="mps", trust_remote_code=True,
    )
    model.eval()
    print("loaded", flush=True)

    for ctx_len in CONTEXT_LENGTHS:
        input_ids = torch.tensor([long_ids[:ctx_len]], dtype=torch.long, device=model.device)
        torch.mps.empty_cache()
        # Leave the last token as a "remainder" for generate()'s own prefill
        # bookkeeping, matching _chunked_prefill_generate's proven pattern --
        # handing generate() a cache already filled to the exact full input
        # length (no remainder) hits an unrelated shape bug in this model's
        # custom cache-position handling.
        prefix_len = ctx_len - 1
        cache_position = torch.arange(0, prefix_len, device=model.device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids[:, :prefix_len], past_key_values=DynamicCache(), use_cache=True,
                cache_position=cache_position,
            )
        cache = outputs.past_key_values

        start = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids, past_key_values=cache,
                max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.time() - start
        n_generated = out.shape[1] - input_ids.shape[1]
        print(f"context={ctx_len}: {elapsed:.2f}s for {n_generated} forced tokens, "
              f"{n_generated/elapsed:.2f} tok/s decode", flush=True)


if __name__ == "__main__":
    main()
