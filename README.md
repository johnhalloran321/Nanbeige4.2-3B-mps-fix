# nanbeige-mps-fix

Fixes for running [Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) via
Hugging Face `transformers` on Apple Silicon (MPS), plus a chunked-prefill fix for its
Looped Transformer memory bound, and every measurement/reproduction script behind
the accompanying paper.

**Patched checkpoint:** [johnhalloran/Nanbeige4.2-3B-mps-fix](https://huggingface.co/johnhalloran/Nanbeige4.2-3B-mps-fix)
**Paper:** arXiv link coming soon
**Long-form writeup:** [docs/index.md](docs/index.md) (same findings, one per section, no compression)

## What's here

| Path | Contents |
|---|---|
| `patch/` | The 5 bug fixes, as a patched `modeling_nanbeige.py` + diff against the original, plus `MPS_FIX_NOTES.md` (bug-by-bug technical writeup) |
| `harness/` | `nanbeige_harness_server.py` — minimal OpenAI-compatible server backing onto the patched checkpoint, chunked prefill included |
| `experiments/` | Every script behind a number in the paper: OOM reproduction, memory/batching sweeps, decode-throughput measurement, MCPMark/BFCL evaluation, the reasoning-loop and path-length ablations |
| `results/` | Committed result JSON (batch sweeps, BFCL, tool schema) and the MCPMark per-task transcripts (`meta.json`/`messages.json`) referenced in the paper |

## Install

```bash
pip install -e .
# only needed for experiments/run_bfcl_benchmark.py:
pip install -e ".[bfcl]"
```

Requires Apple Silicon (MPS) to reproduce the memory/throughput measurements; the
patch itself (`patch/modeling_nanbeige.py`) is not MPS-specific.

## Quick start: run the patched model

```bash
python harness/nanbeige_harness_server.py --port 8100
# pulls johnhalloran/Nanbeige4.2-3B-mps-fix from the Hub by default;
# set NANBEIGE_MODEL_PATH to a local clone to skip re-downloading.
```

```bash
curl http://127.0.0.1:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is 2+2?"}]
}'
```

## Reproducing the paper

Each experiment script is self-contained and documented in its own docstring.
Rough map from paper section to script:

| Paper section | Script(s) |
|---|---|
| §2, Demonstrable bugs | `patch/modeling_nanbeige.patch`, `patch/MPS_FIX_NOTES.md` |
| §3.1, Original OOM / deriving M | `experiments/discover_tool_schema.py` → `experiments/reproduce_original_oom.py` |
| §3.2, Memory/batching sweep (Table 1) | `experiments/prepare_longbench_samples.py` → `experiments/run_batch_sweep.py` |
| §3.3, Decode throughput (Table 2) | `experiments/measure_decode_vs_context.py` |
| §6.2, MCPMark diagnosis | `experiments/reproduce_mcpmark_stall.py`, `experiments/run_mcpmark_isolated.py`, `experiments/diagnose_mps_leak.py`, `experiments/diagnose_stalled_turns.py`, `experiments/run_mcpmark_isolated_shortpath.py` |
| §6.3, BFCL | `experiments/run_bfcl_benchmark.py` |

MCPMark itself ([eval-sys/mcpmark](https://github.com/eval-sys/mcpmark)) is a separate,
third-party benchmark and is not vendored here — clone it yourself and point
`MCPMARK_DIR` at it; see the docstrings in `run_mcpmark_isolated*.py`.

## Headline results

- 5 independent bugs block loading/correct inference out of the box (silently-zeroed
  RoPE buffer, a removed cache API call, a config-dispatch `KeyError`, an MPS-specific
  crash, a `save_pretrained()` failure) — all fixed via sibling-file patching, no
  `transformers` or cached-model-file edits.
- Naive prefill fails via an uncatchable process abort by ~9,000 tokens even at
  batch size 1; chunked prefill remains usable to ~11,000 tokens and supports
  2–4x larger batches at shorter lengths. Neither fixes the original 12,244-token
  production OOM completely.
- Decode throughput falls from 15.5 to 2.1 tokens/sec between a trivial prompt and
  5,120 tokens of context — the actual bottleneck for real multi-turn agentic use,
  not prefill.
- MCPMark (Filesystem, easy tier): 1/10 under its default 600s per-task timeout,
  3/10 once relaxed to its own 3600s default. BFCL (single-turn, isolates
  tool-calling correctness from decode speed): 100% on correctly declining an
  irrelevant call, 63% on a single well-specified call, 3–30% on multiple calls
  in one turn.

Full numbers, methodology, and honest failure-mode diagnosis in the paper (arXiv link coming soon).

## License

Code in this repo is MIT-licensed (see `LICENSE`), except `patch/modeling_nanbeige.py`,
`patch/configuration_nanbeige.py`, and `patch/chat_template.jinja`, which are modified
copies of Nanbeige/Nanbeige4.2-3B's own files and remain under their original
Apache 2.0 license (`patch/LICENSE-APACHE-2.0`).

## Citation

```bibtex
@misc{halloran2026nanbeige,
  title  = {Nanbeige4.2-3B on Apple Silicon: Deployment-Blocking Bugs, a
            Looped-Transformer Memory Bound, and a Partial Chunked-Prefill Fix},
  author = {Halloran, John T.},
  year   = {2026}
}
```
