#!/usr/bin/env python3
"""diagnose_mps_leak.py -- determine whether nanbeige_harness_server's MPS
OOM after several sequential requests is a genuine cross-request memory leak
(torch.mps.empty_cache() not actually reclaiming freed tensors) or simply the
expected, already-documented per-request peak growing with conversation
length (each MCPMark turn resends the whole, growing transcript).

Runs the same _chunked_prefill_generate call repeatedly at a FIXED, modest
prompt length and reports torch.mps.current_allocated_memory() /
driver_allocated_memory() before/after each call and after empty_cache().
If memory keeps climbing across repeated *identical-length* calls, that is a
leak. If it stays flat, the earlier OOM was legitimately caused by
conversation length growing turn over turn.
"""
from __future__ import annotations

import os

import sys

import torch
import transformers
from transformers.cache_utils import DynamicCache

MODEL_PATH = os.environ.get("NANBEIGE_MODEL_PATH", "johnhalloran/Nanbeige4.2-3B-mps-fix")
CHUNK_SIZE = 256
PROMPT_LEN = 2048
N_CALLS = 8


def chunked_prefill_generate(model, input_ids, max_new_tokens, chunk_size=CHUNK_SIZE, **gen_kwargs):
    total_len = input_ids.shape[1]
    if total_len <= chunk_size:
        return model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, **gen_kwargs)
    cache = DynamicCache()
    n_full_chunks = (total_len - 1) // chunk_size
    with torch.no_grad():
        for i in range(n_full_chunks):
            s, e = i * chunk_size, i * chunk_size + chunk_size
            cache_position = torch.arange(s, e, device=input_ids.device)
            outputs = model(input_ids=input_ids[:, s:e], past_key_values=cache,
                             use_cache=True, cache_position=cache_position)
            cache = outputs.past_key_values
    with torch.no_grad():
        return model.generate(input_ids=input_ids, past_key_values=cache,
                               max_new_tokens=max_new_tokens, **gen_kwargs)


def mb(x):
    return x / (1024 ** 2)


def main():
    print(f"loading {MODEL_PATH} ...", flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="mps", trust_remote_code=True,
    )
    model.eval()
    print("loaded", flush=True)

    input_ids = torch.randint(1000, 30000, (1, PROMPT_LEN), device=model.device)

    for i in range(N_CALLS):
        before_alloc = torch.mps.current_allocated_memory()
        before_drv = torch.mps.driver_allocated_memory()
        out = chunked_prefill_generate(model, input_ids, max_new_tokens=16, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id)
        after_alloc = torch.mps.current_allocated_memory()
        after_drv = torch.mps.driver_allocated_memory()
        del out
        torch.mps.empty_cache()
        post_empty_alloc = torch.mps.current_allocated_memory()
        post_empty_drv = torch.mps.driver_allocated_memory()
        print(
            f"call {i}: before(alloc={mb(before_alloc):8.1f}MB drv={mb(before_drv):8.1f}MB) "
            f"after(alloc={mb(after_alloc):8.1f}MB drv={mb(after_drv):8.1f}MB) "
            f"post_empty_cache(alloc={mb(post_empty_alloc):8.1f}MB drv={mb(post_empty_drv):8.1f}MB)",
            flush=True,
        )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
