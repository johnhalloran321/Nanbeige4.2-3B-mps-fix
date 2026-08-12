#!/usr/bin/env python3
"""prepare_longbench_samples.py -- select 50 real LongBench-Pro (Bai et al.)
documents long enough to support truncation to every target length in the
memory/throughput experiment, including M=12244 (the real production-scale
prompt length reproduced in reproduce_original_oom.py), with NO concatenation
of separate documents -- each sample is one real, individually-long context.

Tokenizes each candidate once with the Nanbeige tokenizer (token counts in
the dataset's own "token_length" field are approximate buckets from whatever
tokenizer LongBench-Pro's authors used, not ours) and keeps the first 50
English documents whose real Nanbeige token count is >= MIN_LENGTH.
"""
from __future__ import annotations

import json
import os

import transformers
from huggingface_hub import hf_hub_download

MODEL_PATH = os.environ.get("NANBEIGE_MODEL_PATH", "johnhalloran/Nanbeige4.2-3B-mps-fix")
LONGBENCH_REPO = "caskcsg/LongBench-Pro"
LONGBENCH_FILE = "longbench_pro.json"
MIN_LENGTH = 12244  # M, the largest target length in the sweep
N_SAMPLES = 50
OUT_PATH = os.path.join(os.path.dirname(__file__), "longbench_samples.json")


def main():
    # Behind a TLS-intercepting proxy, set REQUESTS_CA_BUNDLE/SSL_CERT_FILE to
    # your organization's CA bundle before running this -- huggingface_hub
    # respects both.
    longbench_path = hf_hub_download(LONGBENCH_REPO, LONGBENCH_FILE, repo_type="dataset")
    with open(longbench_path) as f:
        data = json.load(f)
    # Prefer larger buckets first so we don't waste tokenizer calls on
    # documents likely too short for MIN_LENGTH.
    bucket_order = {"256k": 0, "128k": 1, "64k": 2, "32k": 3, "16k": 4, "8k": 5}
    candidates = [d for d in data if d["language"] == "English"]
    candidates.sort(key=lambda d: bucket_order.get(d["token_length"], 99))

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    selected = []
    for d in candidates:
        if len(selected) >= N_SAMPLES:
            break
        ids = tokenizer(d["context"])["input_ids"]
        if len(ids) >= MIN_LENGTH:
            selected.append({"id": d["id"], "token_length_bucket": d["token_length"], "input_ids": ids})
            print(f"selected {d['id'][:12]}... bucket={d['token_length']} real_tokens={len(ids)} "
                  f"({len(selected)}/{N_SAMPLES})", flush=True)

    print(f"\nselected {len(selected)} samples, all with >= {MIN_LENGTH} real Nanbeige tokens")
    with open(OUT_PATH, "w") as f:
        json.dump(selected, f)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
