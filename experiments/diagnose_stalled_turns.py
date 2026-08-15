#!/usr/bin/env python3
"""diagnose_stalled_turns.py -- replay the exact conversation state at the
point each of several MCPMark tasks stalled (from the with_fix2_timeout3600_isolated
run), using the real 14-tool filesystem schema, to determine WHY: does the
next turn generate a normal, short, tool-call-terminated response (meaning
the crash logged as "InternalServerError"/"timed out" is purely a resource/
time-budget issue), or does the model run away generating a long, repetitive,
non-terminating response (a genuine reasoning/looping bug, independent of
decode speed)?

Reuses the conversation-reconstruction logic from reproduce_mcpmark_stall.py.
Loads the model once, replays each task's messages.json in turn, with a
generous max_new_tokens (8000) and do_sample=False (matches the harness) so
behavior is deterministic and reproducible.

Supplementary: rules out a reasoning-loop bug as the cause of the paper's
MCPMark failures (Section 5.1, Table 3); not itself a separate paper
section. See docs/index.md for the full writeup.
"""
from __future__ import annotations

import json
import os
import time
import traceback

import torch
import transformers
from transformers.cache_utils import DynamicCache

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get("NANBEIGE_MODEL_PATH", "johnhalloran/Nanbeige4.2-3B-mps-fix")
RESULTS_DIR = os.environ.get(
    "MCPMARK_RESULTS_DIR",
    os.path.join(_REPO_ROOT, "results/mcpmark/with_fix2_3600s_isolated"),
)
SCHEMA_PATH = os.environ.get(
    "TOOL_SCHEMA_PATH", os.path.join(_REPO_ROOT, "results/tool_schema_55.json")
)
TASKS = [
    "file_context__pattern_matching",
    "folder_structure__structure_analysis",
    "papers__papers_counting",
    "student_database__duplicate_name",
    "student_database__recommender_name",
]
MAX_NEW_TOKENS = 8000
CHUNK_SIZE = 256


def _load_filesystem_tools():
    with open(SCHEMA_PATH) as f:
        all_tools = json.load(f)
    fs_tool_names = {
        "read_file", "read_text_file", "read_media_file", "read_multiple_files",
        "write_file", "edit_file", "create_directory", "list_directory",
        "list_directory_with_sizes", "directory_tree", "move_file", "search_files",
        "get_file_info", "list_allowed_directories",
    }
    return [t for t in all_tools if t["function"]["name"] in fs_tool_names]


def _convert_responses_api_to_chat(raw_messages):
    messages = [{"role": "system", "content": (
        "You are a helpful agent that uses tools iteratively to complete the user's task, "
        'and when finished, provides the final answer or simply states "Task completed" without further tool calls.'
    )}]
    pending_tool_calls = []
    for m in raw_messages:
        if m.get("role") == "user":
            messages.append({"role": "user", "content": m["content"]})
        elif m.get("type") == "function_call":
            pending_tool_calls.append({
                "id": m["call_id"], "type": "function",
                "function": {"name": m["name"], "arguments": m["arguments"]},
            })
        elif m.get("type") == "function_call_output":
            if pending_tool_calls:
                messages.append({"role": "assistant", "content": None, "tool_calls": pending_tool_calls})
                pending_tool_calls = []
            messages.append({"role": "tool", "tool_call_id": m["call_id"], "content": m["output"]})
    if pending_tool_calls:
        messages.append({"role": "assistant", "content": None, "tool_calls": pending_tool_calls})
    return messages


def chunked_prefill(model, input_ids, chunk_size=CHUNK_SIZE):
    cache = DynamicCache()
    total_len = input_ids.shape[1]
    n_full_chunks = (total_len - 1) // chunk_size
    with torch.no_grad():
        for i in range(n_full_chunks):
            s, e = i * chunk_size, i * chunk_size + chunk_size
            cache_position = torch.arange(s, e, device=input_ids.device)
            outputs = model(input_ids=input_ids[:, s:e], past_key_values=cache,
                             use_cache=True, cache_position=cache_position)
            cache = outputs.past_key_values
    return cache


def main():
    tools = _load_filesystem_tools()
    print(f"loading tokenizer/model from {MODEL_PATH} ...", flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="mps", trust_remote_code=True,
    )
    model.eval()
    print("loaded\n", flush=True)

    for task in TASKS:
        print(f"\n{'=' * 70}\n=== {task} ===\n{'=' * 70}", flush=True)
        msg_path = f"{RESULTS_DIR}/{task}/messages.json"
        with open(msg_path) as f:
            raw_messages = json.load(f)
        messages = _convert_responses_api_to_chat(raw_messages)

        text = tokenizer.apply_chat_template(
            messages, tools=tools, tool_call_format="json", tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        print(f"prompt_len={prompt_len} tokens, n_messages={len(messages)}", flush=True)

        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            prefill_start = time.time()
            cache = chunked_prefill(model, inputs["input_ids"])
            prefill_elapsed = time.time() - prefill_start
            print(f"chunked prefill done in {prefill_elapsed:.2f}s", flush=True)

            decode_start = time.time()
            with torch.no_grad():
                out = model.generate(
                    input_ids=inputs["input_ids"], past_key_values=cache,
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.pad_token_id,
                )
            decode_elapsed = time.time() - decode_start
            generated_ids = out[0][prompt_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            n_generated = generated_ids.shape[0]
            print(f"GENERATED {n_generated} tokens in {decode_elapsed:.2f}s "
                  f"({n_generated / decode_elapsed:.2f} tok/s), hit_cap={n_generated >= MAX_NEW_TOKENS}",
                  flush=True)
            print("--- FIRST 800 CHARS ---")
            print(generated_text[:800])
            print("--- LAST 800 CHARS ---")
            print(generated_text[-800:])
        except Exception as e:
            print(f"EXCEPTION during replay: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        finally:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
