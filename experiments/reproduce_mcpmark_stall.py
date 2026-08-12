#!/usr/bin/env python3
"""reproduce_mcpmark_stall.py -- replay the exact conversation state of a
real MCPMark task at the point it stalled (turn 4 of file_context__file_splitting,
which completed 3 turns then consumed the rest of its 600s budget with no
response), against the real patched Nanbeige checkpoint, with generation
allowed to run long enough to observe what actually happens: does it stop
cleanly and quickly, or keep generating?

Uses the real filesystem MCP tool schemas (same 14 tools used in the actual
MCPMark run) and the real conversation history reconstructed from
mcpmark/results/.../file_context__file_splitting/messages.json, converted
from litellm's Responses-API-style function_call/function_call_output
messages into standard OpenAI ChatCompletions tool_calls/tool messages (what
our harness's chat-template rendering actually expects).
"""
from __future__ import annotations

import json
import os
import time

import torch
import transformers
from transformers.cache_utils import DynamicCache

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get("NANBEIGE_MODEL_PATH", "johnhalloran/Nanbeige4.2-3B-mps-fix")
MESSAGES_PATH = os.environ.get(
    "MESSAGES_PATH",
    os.path.join(_REPO_ROOT, "results/mcpmark/with_fix2_600s/file_context__file_splitting/messages.json"),
)
SCHEMA_PATH = os.environ.get("TOOL_SCHEMA_PATH", os.path.join(_REPO_ROOT, "results/tool_schema_55.json"))


def _load_filesystem_tools():
    with open(SCHEMA_PATH) as f:
        all_tools = json.load(f)
    fs_tool_names = {
        "read_file", "read_text_file", "read_media_file", "read_multiple_files",
        "write_file", "edit_file", "create_directory", "list_directory",
        "list_directory_with_sizes", "directory_tree", "move_file", "search_files",
        "get_file_info", "list_allowed_directories",
    }
    fs_tools = [t for t in all_tools if t["function"]["name"] in fs_tool_names]
    print(f"filesystem tools: {len(fs_tools)}")
    return fs_tools


def _convert_responses_api_to_chat(raw_messages):
    """Convert litellm Responses-API-style function_call/function_call_output
    entries into standard OpenAI ChatCompletions messages."""
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


def main():
    with open(MESSAGES_PATH) as f:
        raw_messages = json.load(f)
    messages = _convert_responses_api_to_chat(raw_messages)
    tools = _load_filesystem_tools()

    print(f"loading tokenizer/model from {MODEL_PATH} ...", flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="mps", trust_remote_code=True,
    )
    model.eval()

    text = tokenizer.apply_chat_template(
        messages, tools=tools, tool_call_format="json", tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    print(f"\n*** RECONSTRUCTED STALLED-TURN PROMPT LENGTH = {prompt_len} tokens ***\n", flush=True)

    # Chunked prefill (matches the harness), then decode with a generous but
    # bounded token budget so we can directly observe what the model does
    # instead of waiting for the full uncapped max_tokens=32768.
    chunk_size = 256
    cache = DynamicCache()
    total_len = prompt_len
    n_full_chunks = (total_len - 1) // chunk_size
    prefill_start = time.time()
    with torch.no_grad():
        for i in range(n_full_chunks):
            s, e = i * chunk_size, i * chunk_size + chunk_size
            cache_position = torch.arange(s, e, device=inputs["input_ids"].device)
            outputs = model(
                input_ids=inputs["input_ids"][:, s:e], past_key_values=cache,
                use_cache=True, cache_position=cache_position,
            )
            cache = outputs.past_key_values
    print(f"chunked prefill done in {time.time() - prefill_start:.2f}s", flush=True)

    print("generating (up to 3000 tokens, watching for early stop) ...", flush=True)
    decode_start = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"], past_key_values=cache,
            max_new_tokens=3000, do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    decode_elapsed = time.time() - decode_start
    generated_ids = out[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    n_generated = generated_ids.shape[0]
    print(f"\n*** GENERATED {n_generated} TOKENS IN {decode_elapsed:.2f}s "
          f"({n_generated/decode_elapsed:.2f} tok/s) ***", flush=True)
    print(f"hit_max_new_tokens={n_generated >= 3000}", flush=True)
    print("\n--- FIRST 1000 CHARS OF GENERATED TEXT ---")
    print(generated_text[:1000])
    print("\n--- LAST 1000 CHARS OF GENERATED TEXT ---")
    print(generated_text[-1000:])


if __name__ == "__main__":
    main()
