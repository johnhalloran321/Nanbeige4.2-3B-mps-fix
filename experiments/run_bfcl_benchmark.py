#!/usr/bin/env python3
"""run_bfcl_benchmark.py -- evaluate the Nanbeige harness server on a subset
of the Berkeley Function-Calling Leaderboard (BFCL v4), using categories that
are single-turn, deterministic (AST/exact-match), and require no external
LLM judge: simple_python, multiple, parallel, parallel_multiple, irrelevance.

Motivation: MCPMark's near-zero success rate (Table 3) is dominated by an
already-diagnosed, unrelated decode-throughput bottleneck (Section 3.3) --
each MCPMark task requires many sequential turns with large uncapped
generation budgets, so wall-clock timeouts are hit before task logic is
exercised at all. BFCL's non-live, non-multi-turn categories are one
generation per task with a small output (a single tool call), which isolates
tool-calling *correctness* from that confound.

Requires: `pip install bfcl-eval soundfile` (soundfile only needed to satisfy
an import chain inside bfcl_eval.constants.model_config -- unrelated to
audio, never actually used by the AST checkers this script imports) and the
nanbeige_harness_server.py running on http://127.0.0.1:8100.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import requests

from bfcl_eval.constants.enums import Language
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

BFCL_DATA_DIR = None  # resolved in main()
SERVER_URL = "http://127.0.0.1:8100/v1/chat/completions"
MODEL_NAME_FOR_CHECKER = "gpt-4o-2024-11-20"  # passthrough; only used for
# convert_func_name's underscore_to_dot lookup, irrelevant to our own model.

CATEGORIES = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance"]
N_PER_CATEGORY = 30

_BFCL_TYPE_TO_JSON_SCHEMA = {
    "string": "string",
    "integer": "integer",
    "float": "number",
    "boolean": "boolean",
    "array": "array",
    "tuple": "array",
    "dict": "object",
    "any": "string",
}


def _convert_param_schema(param: dict) -> dict:
    out = dict(param)
    bfcl_type = out.get("type")
    if bfcl_type in _BFCL_TYPE_TO_JSON_SCHEMA:
        out["type"] = _BFCL_TYPE_TO_JSON_SCHEMA[bfcl_type]
    if "items" in out and isinstance(out["items"], dict):
        out["items"] = _convert_param_schema(out["items"])
    if "properties" in out and isinstance(out["properties"], dict):
        out["properties"] = {k: _convert_param_schema(v) for k, v in out["properties"].items()}
    return out


def bfcl_function_to_openai_tool(func_description: dict) -> dict:
    params = func_description.get("parameters", {"type": "dict", "properties": {}, "required": []})
    return {
        "type": "function",
        "function": {
            "name": func_description["name"],
            "description": func_description.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": {
                    k: _convert_param_schema(v) for k, v in params.get("properties", {}).items()
                },
                "required": params.get("required", []),
            },
        },
    }


def load_category(name: str, n: int):
    q_path = os.path.join(BFCL_DATA_DIR, f"BFCL_v4_{name}.json")
    with open(q_path) as f:
        questions = [json.loads(line) for line in f][:n]
    possible_answer = None
    if name != "irrelevance":
        pa_path = os.path.join(BFCL_DATA_DIR, "possible_answer", f"BFCL_v4_{name}.json")
        with open(pa_path) as f:
            pa_by_id = {}
            for line in f:
                obj = json.loads(line)
                pa_by_id[obj["id"]] = obj["ground_truth"]
        possible_answer = pa_by_id
    return questions, possible_answer


def call_model(question: dict, timeout=120) -> tuple[list, float]:
    tools = [bfcl_function_to_openai_tool(fn) for fn in question["function"]]
    messages = []
    for turn in question["question"]:
        for msg in turn:
            messages.append({"role": msg["role"], "content": msg["content"]})
    payload = {"model": "nanbeige42-harness", "messages": messages, "tools": tools, "max_tokens": 256}
    start = time.time()
    resp = requests.post(SERVER_URL, json=payload, timeout=timeout)
    elapsed = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    tool_calls = data["choices"][0]["message"].get("tool_calls") or []
    return tool_calls, elapsed


def tool_calls_to_model_output(tool_calls: list) -> list[dict]:
    """Convert OpenAI-style tool_calls into BFCL's expected model_output
    shape: a list of {func_name: {param: value, ...}} dicts, one per call."""
    out = []
    for tc in tool_calls:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            args = {}
        out.append({name: args})
    return out


def evaluate_category(name: str, n: int):
    questions, possible_answer = load_category(name, n)
    results = []
    for q in questions:
        try:
            tool_calls, elapsed = call_model(q)
        except Exception as e:
            results.append({"id": q["id"], "valid": False, "error": f"request_failed: {e}"})
            continue

        if name == "irrelevance":
            valid = len(tool_calls) == 0
            results.append({"id": q["id"], "valid": valid, "n_tool_calls": len(tool_calls), "elapsed": elapsed})
            continue

        model_output = tool_calls_to_model_output(tool_calls)
        gt = possible_answer[q["id"]]
        if name in ("simple_python",):
            if len(model_output) != 1:
                results.append({"id": q["id"], "valid": False, "error": "wrong_call_count", "elapsed": elapsed})
                continue
            verdict = ast_checker(
                q["function"], model_output, gt, Language.PYTHON, "simple", MODEL_NAME_FOR_CHECKER,
            )
        elif name == "multiple":
            if len(model_output) != 1:
                results.append({"id": q["id"], "valid": False, "error": "wrong_call_count", "elapsed": elapsed})
                continue
            verdict = ast_checker(
                q["function"], model_output, gt, Language.PYTHON, "multiple", MODEL_NAME_FOR_CHECKER,
            )
        else:  # parallel, parallel_multiple
            verdict = ast_checker(
                q["function"], model_output, gt, Language.PYTHON, name, MODEL_NAME_FOR_CHECKER,
            )
        results.append({"id": q["id"], "valid": verdict.get("valid", False),
                         "error": verdict.get("error"), "elapsed": elapsed})
    return results


def main():
    global BFCL_DATA_DIR
    import bfcl_eval
    BFCL_DATA_DIR = os.path.join(os.path.dirname(bfcl_eval.__file__), "data")

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_PER_CATEGORY)
    parser.add_argument("--categories", default=",".join(CATEGORIES))
    parser.add_argument("--out", default="bfcl_results.json")
    args = parser.parse_args()

    all_results = {}
    for cat in args.categories.split(","):
        print(f"=== {cat} ===", flush=True)
        results = evaluate_category(cat, args.n)
        n_valid = sum(1 for r in results if r["valid"])
        print(f"{cat}: {n_valid}/{len(results)} = {100 * n_valid / len(results):.1f}%", flush=True)
        all_results[cat] = results

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Summary ===")
    for cat, results in all_results.items():
        n_valid = sum(1 for r in results if r["valid"])
        print(f"{cat:20s} {n_valid:3d}/{len(results):3d} = {100 * n_valid / len(results):5.1f}%")


if __name__ == "__main__":
    main()
