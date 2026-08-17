# nanbeige-mps-fix

Fixes for running [Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) via
Hugging Face `transformers` on Apple Silicon (MPS), plus a chunked-prefill fix for its
Looped Transformer memory bound, and every measurement/reproduction script behind
the accompanying paper.

**Patched checkpoint:** [johnhalloran/Nanbeige4.2-3B-mps-fix](https://huggingface.co/johnhalloran/Nanbeige4.2-3B-mps-fix)
(5 core bugs only — see [below](#what-the-hf-checkpoint-does-and-doesnt-include) for what needs this repo)
**Paper:** [arXiv:2608.13987](https://arxiv.org/abs/2608.13987)
**Long-form writeup:** [docs/index.md](docs/index.md) (same findings, one per section, no compression)

## What's here

| Path | Contents |
|---|---|
| `patch/` | The 5 bug fixes, as a patched `modeling_nanbeige.py` + diff against the original, plus `MPS_FIX_NOTES.md` (bug-by-bug technical writeup) |
| `harness/` | `nanbeige_harness_server.py` — minimal OpenAI-compatible server backing onto the patched checkpoint, chunked prefill included |
| `experiments/` | Every script behind the paper's tables, plus supplementary experiments (OOM reproduction, decode-throughput measurement, the stalled-turn and path-length diagnostics) referenced in its prose or covered in `docs/index.md` |
| `results/` | Committed result JSON (batch sweeps, BFCL, tool schema) and the MCPMark per-task transcripts (`meta.json`/`messages.json`) referenced in the paper |

## Install

```bash
pip install -e .
# only needed for experiments/run_bfcl_benchmark.py:
pip install -e ".[bfcl]"
```

Requires Apple Silicon (MPS) to reproduce the memory/throughput measurements; the
patch itself (`patch/modeling_nanbeige.py`) is not MPS-specific.

## What the HF checkpoint does and doesn't include

The five bugs in `patch/` (RoPE buffer zeroing, the RoPE-config dispatch `KeyError`,
the removed cache-API call, the MPS position-IDs crash, `save_pretrained`) are baked
into [`johnhalloran/Nanbeige4.2-3B-mps-fix`](https://huggingface.co/johnhalloran/Nanbeige4.2-3B-mps-fix)'s
weights/config directly — loading it via plain `transformers` gets you those fixes for
free, no code from this repo required.

The other two fixes are **not** on the checkpoint, because neither one can be baked in:

- **Chunked prefill** (Section 3.1) is a serving-time strategy, not a weight or config
  change — it only exists in `harness/nanbeige_harness_server.py`.
- **The system-prompt splice** (Section 4) patches the chat template's *rendered
  output*, not the template file itself — the HF checkpoint's `chat_template.jinja` is
  byte-identical to the unpatched original. Loading the checkpoint directly and
  supplying any system message (as most agent frameworks do) still triggers the
  regression.

If you load the HF checkpoint directly instead of going through this repo's harness,
you'll still hit the memory ceiling and the system-prompt regression — use
`harness/nanbeige_harness_server.py`, or port the relevant fix from
`harness/nanbeige_harness_server.py` / `patch/MPS_FIX_NOTES.md` yourself.

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
Map from paper section to script:

| Paper section | Script(s) |
|---|---|
| §2, Five Initial Deployment Bugs | `patch/modeling_nanbeige.patch`, `patch/MPS_FIX_NOTES.md` |
| §3.1, Chunked-prefill method | `patch/` fix, `_chunked_prefill_generate` in `harness/nanbeige_harness_server.py` |
| §3.2, LongBench-Pro memory/batching sweep (Table 2) | `experiments/prepare_longbench_samples.py` → `experiments/run_batch_sweep.py` |
| §4, System-prompt regression | `patch/chat_template.jinja`, `patch/MPS_FIX_NOTES.md` |
| §5, MPS memory bug + per-task server isolation | `experiments/diagnose_mps_leak.py`, `experiments/run_mcpmark_isolated.py` |
| §5.1, MCPMark, Filesystem easy tier (Table 3) | `experiments/run_mcpmark_isolated.py` |
| §5.2, BFCL (Table 4) | `experiments/run_bfcl_benchmark.py` |

MCPMark itself ([eval-sys/mcpmark](https://github.com/eval-sys/mcpmark)) is a separate,
third-party benchmark and is not vendored here — clone it yourself and point
`MCPMARK_DIR` at it; see the docstrings in `run_mcpmark_isolated*.py`.

### Supplementary experiments (not tabulated in the paper)

These aren't behind a numbered section/table in `paper.tex` — they underpin
claims made in its prose, motivated later experiments, or were cut for space
during condensing to the arXiv version. All are covered in full in
[docs/index.md](docs/index.md).

| What it establishes | Script(s) |
|---|---|
| Deriving $M=12{,}244$, the production-OOM token length used as the largest length in Table 2 | `experiments/discover_tool_schema.py` → `experiments/reproduce_original_oom.py` |
| Decode throughput vs. context length — motivates isolating tool-calling correctness from decode speed via BFCL (§5.2) | `experiments/measure_decode_throughput.py`, `experiments/measure_decode_vs_context.py` |
| Why individual MCPMark tasks stalled: resource/timeout exhaustion vs. a genuine reasoning-loop bug | `experiments/reproduce_mcpmark_stall.py`, `experiments/diagnose_stalled_turns.py` |
| Path-length ablation: does a shorter absolute working-directory path change MCPMark outcomes? | `experiments/run_mcpmark_isolated_shortpath.py` |

## Headline results

- 5 independent bugs block loading/correct inference out of the box (silently-zeroed
  RoPE buffer, a removed cache API call, a config-dispatch `KeyError`, an MPS-specific
  crash, a `save_pretrained()` failure) — all fixed via sibling-file patching, no
  `transformers` or cached-model-file edits. (§2)
- Naive prefill fails via an uncatchable process abort by ~8,000 tokens even at
  batch size 1; chunked prefill remains usable to ~11,000 tokens and supports
  2–4x larger batches at shorter lengths. Neither fixes the original 12,244-token
  production OOM completely. (§3.2, Table 2)
- MCPMark (Filesystem, easy tier): 3/10 under its own default 3600s per-task
  timeout. (§5.1, Table 3)
- BFCL (single-turn, isolates tool-calling correctness from decode speed): 100% on
  correctly declining an irrelevant call, 63% on a single well-specified call,
  3–30% on multiple calls in one turn. (§5.2, Table 4)

Full numbers and methodology in the [paper](https://arxiv.org/abs/2608.13987).

### Supplementary findings (not in the paper)

- Decode throughput falls from 15.5 to 2.1 tokens/sec between a trivial prompt and
  5,120 tokens of context — the actual bottleneck for real multi-turn agentic use,
  not prefill. See `experiments/measure_decode_vs_context.py`.
- Under a stricter 600s per-task timeout (vs. the paper's 3600s), MCPMark drops to
  1/10 — an earlier run kept for comparison (`results/mcpmark/with_fix2_600s`).

The honest failure-mode diagnosis behind these is in `docs/index.md`.

## License

Code in this repo is MIT-licensed (see `LICENSE`), except `patch/modeling_nanbeige.py`,
`patch/configuration_nanbeige.py`, and `patch/chat_template.jinja`, which are modified
copies of Nanbeige/Nanbeige4.2-3B's own files and remain under their original
Apache 2.0 license (`patch/LICENSE-APACHE-2.0`).

## Citation

```bibtex
@misc{halloran2026nanbeige,
  title  = {Nanbeige4.2-3B on Apple Silicon: Fixing Deployment Bugs and
Decreasing Looped Transformer Memory Overhead},
  author = {Halloran, John T.},
  year   = {2026}
}
```
