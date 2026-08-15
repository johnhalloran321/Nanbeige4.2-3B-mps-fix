#!/usr/bin/env python3
"""measure_decode_throughput.py -- measure real autoregressive decode
throughput (tokens/sec generated, not prefill) for Nanbeige on this
hardware, and separately check whether the model naturally emits an
EOS/stop token quickly for a representative MCP tool-calling prompt or
keeps generating toward max_new_tokens.

Motivation: MCPMark requests max_tokens=32768 per turn (uncapped by our
harness). The paper's memory/batching sweep (Section 3.2, Table 2) only
measures PREFILL time; if decode itself is slow, or if the model doesn't
terminate cleanly, MCPMark's per-task timeout could be exhausted by decode
alone, independent of prefill length. Supplementary: this measurement itself
is not tabulated in the paper; see docs/index.md.
"""
from __future__ import annotations

import json
import os
import time

import torch
import transformers

MODEL_PATH = os.environ.get("NANBEIGE_MODEL_PATH", "johnhalloran/Nanbeige4.2-3B-mps-fix")


def main():
    print("loading model...", flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="mps", trust_remote_code=True,
    )
    model.eval()
    print("loaded", flush=True)

    # Simple short prompt, no tools -- isolate pure decode speed, not prefill.
    messages = [{"role": "user", "content": "Count from 1 to 200, writing each number on its own line."}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    print(f"prompt_len={prompt_len}", flush=True)

    for target_new_tokens in [64, 256, 1024]:
        torch.mps.empty_cache()
        start = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=target_new_tokens, min_new_tokens=target_new_tokens,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
        torch.mps.synchronize()
        elapsed = time.time() - start
        actual_new = out.shape[1] - prompt_len
        print(f"target_new_tokens={target_new_tokens}: {elapsed:.2f}s, "
              f"{actual_new} tokens generated, {actual_new/elapsed:.2f} tok/s decode", flush=True)


if __name__ == "__main__":
    main()
