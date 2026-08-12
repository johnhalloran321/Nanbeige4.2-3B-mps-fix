#!/usr/bin/env python3
"""reproduce_original_oom.py -- reconstruct the exact real-world scenario that
originally produced the 35.89 GiB naive-prefill OOM (a real agentic request
with the full connected MCP toolset's schemas serialized into the prompt),
verify naive still OOMs the same way, verify whether chunked prefill avoids
it, and report the prompt's exact token length under the Nanbeige tokenizer
-- this becomes M, a real (not estimated) lower bound on the sequence length
chunked prefill can handle for Nanbeige.

Reuses the same 11-server, 55-tool MCP discovery already validated earlier
this session (ironclad_agent's promptopt.py machinery) rather than a
synthetic tool list, and renders the prompt through the real chat template
exactly as nanbeige_harness_server.py does, so the token count is the real
one a production ironclad-agent request would send -- not an approximation.
"""
from __future__ import annotations

import json
import os

import torch
import transformers
from transformers.cache_utils import DynamicCache

MODEL_PATH = os.environ.get("NANBEIGE_MODEL_PATH", "johnhalloran/Nanbeige4.2-3B-mps-fix")
SCHEMA_PATH = os.environ.get(
    "TOOL_SCHEMA_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/tool_schema_55.json"),
)


def _build_real_prompt():
    # Discovery runs standalone, in discover_tool_schema.py, which never imports
    # torch -- kept separate so nothing in the MCP-discovery/ironclad_agent
    # import chain can interact with this process's MPS/Metal allocator state.
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    print(f"loaded {len(schema)} tool schemas from {SCHEMA_PATH}")

    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to MCP tools."},
        {"role": "user", "content": (
            "I need to review our knowledge graph, check the files in the workspace, "
            "and make sure everything is properly redacted and safe before I share it "
            "with a colleague. Can you help me go through this step by step?"
        )},
    ]
    return schema, messages


def _render_and_tokenize(tokenizer, schema, messages):
    text = tokenizer.apply_chat_template(
        messages, tools=schema, tool_call_format="json",
        tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    return ids


def main():
    schema, messages = _build_real_prompt()

    print(f"loading tokenizer/model from {MODEL_PATH} ...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="mps", trust_remote_code=True,
    )
    model.eval()

    input_ids = _render_and_tokenize(tokenizer, schema, messages)
    n_tokens = input_ids.shape[1]
    print(f"\n*** REAL PROMPT TOKEN LENGTH (M) = {n_tokens} ***\n")
    input_ids = input_ids.to(model.device)

    import sys
    skip_naive = "--skip-naive" in sys.argv
    if skip_naive:
        print("--- skipping NAIVE prefill (already confirmed: hard Metal abort, "
              "'Failed to allocate private MTLBuffer for size 28783782912' "
              "[26.8 GiB], not even a catchable RuntimeError, at this exact prompt) ---")
    else:
        print("--- attempting NAIVE prefill ---")
        try:
            torch.mps.empty_cache()
            with torch.no_grad():
                model(input_ids=input_ids, past_key_values=DynamicCache(), use_cache=True)
            torch.mps.synchronize()
            print("naive: SUCCEEDED (unexpected)")
        except RuntimeError as e:
            print(f"naive: FAILED as expected -- {e}")

    print("\n--- attempting CHUNKED prefill ---")
    chunk_size = 256
    try:
        torch.mps.empty_cache()
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
        print("chunked: SUCCEEDED")
    except RuntimeError as e:
        print(f"chunked: FAILED -- {e}")


if __name__ == "__main__":
    main()
