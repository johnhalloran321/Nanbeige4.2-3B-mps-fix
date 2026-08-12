#!/usr/bin/env python3
"""
nanbeige_harness_server.py — a minimal OpenAI-compatible /v1/chat/completions
server, backing directly onto a plain `model.generate()` call.

NOT official support. This exists because ironclad-agent's ReAct loop only
talks to Ollama, vLLM, or Azure/OpenAI-compatible backends (agent.py's
`openai.AsyncOpenAI`/`ollama.AsyncClient` dispatch) -- there's no in-process
transformers path for the model that drives tool-calling. Standing up real
vLLM serving for Nanbeige turned out to need a custom vLLM model
implementation (its looped/44-virtual-layer KV cache has no vLLM paged-
attention equivalent, and its attention classes predate vLLM's
ALL_ATTENTION_FUNCTIONS integration point) -- a real engineering project, not
a patch. This sidesteps all of that: ironclad-agent already knows how to
speak the OpenAI chat-completions HTTP contract, so this just implements the
minimal slice of that contract, backed by a synchronous `generate()` call
against the already-fixed checkpoint (see ../patch/MPS_FIX_NOTES.md for the
bug history). No vLLM, no Ollama, no paged attention -- one request in, one
generate() call, one response out.

Defaults to pulling the patched checkpoint from the Hub
(johnhalloran/Nanbeige4.2-3B-mps-fix); set NANBEIGE_MODEL_PATH to a local
directory (e.g. a `git clone` of that same repo, or ../patch plus the
weights/tokenizer files copied in) to avoid re-downloading on every run.

Usage:
    python nanbeige_harness_server.py [--port 8100] [--max-new-tokens 512]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid

import torch
import transformers
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers.cache_utils import DynamicCache

MODEL_PATH = os.environ.get(
    "NANBEIGE_MODEL_PATH",
    "johnhalloran/Nanbeige4.2-3B-mps-fix",
)
SERVED_MODEL_NAME = os.environ.get("NANBEIGE_SERVED_NAME", "nanbeige42-harness")
# Chunked prefill: this checkpoint's looped design means every attention layer
# effectively runs 44 times (num_loops=2 x 22 physical layers). A normal
# single-shot prefill materializes a full (prompt_len x prompt_len) attention
# score matrix per effective layer -- with ~20+ MCP tool schemas serialized
# into the prompt, that alone was enough to OOM MPS (35.89 GiB observed for
# one request). Processing the prompt in fixed-size chunks, growing the KV
# cache between them via the same incremental mechanism normal decoding
# already uses, bounds the per-step attention matrix to (chunk x
# running-total) instead of (prompt_len x prompt_len) -- verified bit-
# identical output against plain generate() on a short prompt before relying
# on it here. See MPS_FIX_NOTES.md for the OOM root-cause writeup.
PREFILL_CHUNK_SIZE = int(os.environ.get("NANBEIGE_PREFILL_CHUNK_SIZE", "256"))

# Ablation-only toggle for the paper's before/after sys-prompt comparison --
# not something a real deployment would ever want off. When set, skips
# _extract_leading_system_content/_splice_system_content entirely so the
# caller's system message goes straight through the template's normal (buggy)
# replace-not-append branch, reproducing the pre-fix behavior on demand
# without needing a second copy of this server.
DISABLE_SYSPROMPT_FIX = os.environ.get("NANBEIGE_DISABLE_SYSPROMPT_FIX", "") == "1"

# chat_template.jinja:20-52 (the `tools` branch) does a straight if/else on
# messages[0]: if the caller supplies ANY system message, it's used verbatim
# (with a trailing "\n\n" the template adds) -- otherwise the template
# injects this exact default preamble (Nanbeige's own trained-in tool-use
# system prompt: "You are a tool-function-calling expert...") with NO
# trailing separator before "# Tools". It's a replace, not an append/merge,
# AND the two branches aren't whitespace-symmetric.
#
# ironclad-agent's _base_system_prompt (agent.py:278) always supplies its own
# system message, which silently discards this default -- confirmed
# directly: the same tools+user-message request produces a clean, correctly-
# formatted single tool call with no caller system message, and garbled,
# malformed multi-tool-call syntax the instant ANY caller system message is
# added. Simply prepending this text as message content does NOT fix it --
# confirmed by testing byte-identical content supplied as an explicit system
# message: it still breaks, because the template's own "\n\n" it adds after
# any explicit system message differs from the zero-separator auto-insert
# path. That difference alone (2 characters) was enough to flip clean
# single-tool-call output into unbounded, never-converging deliberation.
# This checkpoint's tool-calling reliability is evidently calibrated to the
# EXACT byte sequence its own rendering path produces, down to whitespace --
# plausibly because its tool-use SFT/RL data was only ever rendered through
# the auto-insert branch, never with a caller-supplied system message.
#
# Fix: render with NO system message at all (so the template uses its own
# zero-extra-whitespace auto-insert), then splice the caller's system content
# into the rendered string directly -- never going through the template's
# asymmetric system-message branch. Verified this restores clean,
# single-tool-call output with the caller's own system content included. See
# MPS_FIX_NOTES.md.
_NANBEIGE_DEFAULT_TOOL_SYSTEM_PROMPT = (
    "你是一位工具函数调用专家，你会得到一个问题和一组可能的工具函数。根据问题，你需要进行一个或多个函数/工具调用以实现目的，请尽量尝试探索通过工具解决问题。\n"
    "如果没有一个函数可以使用，请直接使用自然语言回复用户。\n"
    "如果给定的问题缺少函数所需的参数，请使用自然语言进行提问，向用户询问必要信息。\n"
    "如果调用结果已经足够回答用户问题，请对历史结果进行总结，使用自然语言回复用户。"
)


def _extract_leading_system_content(messages: list[dict]) -> tuple[list[dict], str | None]:
    """Strip a leading system message out of `messages` (so templating uses
    the auto-insert default), returning (remaining_messages, system_content)."""
    if messages and messages[0].get("role") == "system":
        return messages[1:], messages[0].get("content", "")
    return messages, None


def _splice_system_content(rendered: str, system_content: str | None) -> str:
    """Insert caller-supplied system content right after the auto-inserted
    default preamble, matching its exact (zero-extra-newline) formatting
    instead of going through the template's own system-message branch.

    ironclad-agent's ModelProfile.default_tool_system_prompt mechanism
    (config.py) now also prepends this same default text to the system
    message it builds -- a best-effort mitigation on its side, since it
    doesn't control chat-template rendering the way this wrapper does. If
    that already happened, strip the duplicate before splicing so it isn't
    inserted twice; the wrapper's own byte-exact splice below is what
    actually matters here regardless of what ironclad-agent already tried.
    """
    if not system_content:
        return rendered
    if system_content.startswith(_NANBEIGE_DEFAULT_TOOL_SYSTEM_PROMPT):
        system_content = system_content[len(_NANBEIGE_DEFAULT_TOOL_SYSTEM_PROMPT):].lstrip("\n")
    if not system_content:
        return rendered
    marker = _NANBEIGE_DEFAULT_TOOL_SYSTEM_PROMPT + "# Tools"
    replacement = _NANBEIGE_DEFAULT_TOOL_SYSTEM_PROMPT + "\n" + system_content + "\n# Tools"
    if marker not in rendered:
        # No tools in this request (auto-insert default only fires when tools
        # are present) -- nothing to splice into, fall back to leaving it out
        # rather than guessing at a different insertion point.
        return rendered
    return rendered.replace(marker, replacement, 1)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Observed variant: a bare function name with no JSON at all (no-argument
# calls, e.g. `<tool_call>read_graph</tool_call>` instead of
# `<tool_call>{"name": "read_graph", "arguments": {}}</tool_call>`) -- the
# model's own tool_call_format='json' convention isn't perfectly consistent
# about this. Matched separately, only against text the JSON regex didn't
# already claim.
_BARE_TOOL_CALL_RE = re.compile(r"<tool_call>\s*([A-Za-z_][\w\-]*)\s*</tool_call>")

app = FastAPI()
_tokenizer = None
_model = None


def _load() -> None:
    global _tokenizer, _model
    if _model is not None:
        return
    print(f"loading {MODEL_PATH} ...")
    _tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    _model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    print(f"loaded on {device}")


def _chunked_prefill_generate(model, input_ids, max_new_tokens, chunk_size=PREFILL_CHUNK_SIZE, **gen_kwargs):
    """Prefill in fixed-size chunks (growing the KV cache incrementally)
    instead of one single-shot forward pass over the whole prompt, then hand
    off the remainder + decoding to the model's own generate(). Bit-identical
    to plain generate() (verified), just bounded peak attention memory."""
    total_len = input_ids.shape[1]
    if total_len <= chunk_size:
        return model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, **gen_kwargs)

    cache = DynamicCache()
    n_full_chunks = (total_len - 1) // chunk_size  # leave the remainder for generate()'s own path
    with torch.no_grad():
        for i in range(n_full_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = input_ids[:, start:end]
            cache_position = torch.arange(start, end, device=input_ids.device)
            outputs = model(
                input_ids=chunk,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
            cache = outputs.past_key_values

    with torch.no_grad():
        return model.generate(
            input_ids=input_ids,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )


def _extract_tool_calls(text: str) -> tuple[str | None, list[dict]]:
    """Pull <tool_call>...</tool_call> blocks (this checkpoint's own
    tool_call_format='json' convention -- see chat_template.jinja) out of the
    generated text, returning (remaining_content_or_None, tool_calls)."""
    tool_calls = []
    claimed_spans = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        claimed_spans.append(match.span())
        arguments = obj.get("arguments", {})
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": obj.get("name", ""),
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                },
            }
        )
    for match in _BARE_TOOL_CALL_RE.finditer(text):
        if any(start <= match.start() < end for start, end in claimed_spans):
            continue
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": match.group(1), "arguments": "{}"},
            }
        )
    if tool_calls:
        return None, tool_calls
    return text, []


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _load()
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools") or None
    max_new_tokens = body.get("max_tokens") or int(os.environ.get("NANBEIGE_MAX_NEW_TOKENS", "512"))

    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    system_content = None
    if tools and not DISABLE_SYSPROMPT_FIX:
        template_kwargs["tools"] = tools
        template_kwargs["tool_call_format"] = "json"
        # Only strip+splice when tools are in play -- the asymmetric-
        # whitespace auto-insert default this works around only exists
        # inside the template's `{% if tools %}` branch in the first place.
        messages, system_content = _extract_leading_system_content(messages)
    elif tools:
        template_kwargs["tools"] = tools
        template_kwargs["tool_call_format"] = "json"
    text = _tokenizer.apply_chat_template(messages, **template_kwargs)
    if not DISABLE_SYSPROMPT_FIX:
        text = _splice_system_content(text, system_content)
    inputs = _tokenizer([text], return_tensors="pt").to(_model.device)

    try:
        output_ids = _chunked_prefill_generate(
            _model,
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_tokenizer.pad_token_id,
        )
    finally:
        # Each request builds its own DynamicCache; dropping the Python
        # reference doesn't return the underlying device memory on MPS,
        # whose caching allocator retains freed blocks for reuse rather than
        # releasing them. Across many sequential requests in a long-lived
        # process (e.g. a multi-turn agentic conversation with growing
        # context per turn), that retained-but-unused memory compounds and
        # can OOM even though chunked prefill bounds any single request's
        # peak allocation -- observed directly during a multi-turn MCPMark
        # replay (turn 5 OOM'd at 32.78 GiB resident with nothing else
        # running). Freeing after every request keeps steady-state memory
        # bounded by the current request instead of the request history.
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    generated = _tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )
    content, tool_calls = _extract_tool_calls(generated)

    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls

    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model") or SERVED_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(output_ids.shape[1] - inputs["input_ids"].shape[1]),
            "total_tokens": int(output_ids.shape[1]),
        },
    }
    return JSONResponse(response)


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    _load()
    uvicorn.run(app, host=args.host, port=args.port)
