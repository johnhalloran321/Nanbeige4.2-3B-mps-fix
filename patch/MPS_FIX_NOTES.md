# Nanbeige4.2-3B: five `transformers`-compatibility bugs, found and fixed

This repo is a derivative of [`Nanbeige/Nanbeige4.2-3B`](https://huggingface.co/Nanbeige/Nanbeige4.2-3B)
(base revision `5d54321e9e01e0d026f8e371046678fc384dca39`, Apache-2.0), with five
bugs fixed in the custom `modeling_nanbeige.py`/`configuration_nanbeige.py` and
in the checkpoint's own weights. It is **not affiliated with or endorsed by the
Nanbeige team**. All credit for the model architecture, training, and weights
belongs to them.

**Why this exists:** running the stock checkpoint through generic HF
`transformers` (`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`)
crashes hard on Apple Silicon (MPS) after a few generation steps, and — even
once that crash is patched around — produces structurally incoherent output
(looping `</think>` tokens, character-level word salad) regardless of device,
dtype, or sampling settings. Five distinct bugs are responsible, all
compatibility issues between the model's own custom code and the installed
`transformers` version (`5.8.1` at time of writing), not anything wrong with
the model's weights or training. None are security issues — the custom code
was reviewed by hand before any of this started (no `eval`/`exec`/`subprocess`/
`pickle`/network calls beyond the expected Hub interactions).

## Architecture reality check

Before the bug list: it's worth being explicit about what this checkpoint
*actually* runs, versus what the architecture is capable of. `configuration_nanbeige.py`
defines a rich set of options — LoopSplit (`enable_double_loop_split`),
manifold-constrained hyper-connections / mHC (`enable_mhc`, `enable_hyper_connection`,
Sinkhorn-normalized residual mixing), depth attention (`enable_depth_attention`),
and n-gram embeddings (`emb_neighbor_num` and friends). **None of them are
enabled by this checkpoint's `config.json`** — every one of those fields is
either absent (defaulting to `False`/`None`) or explicitly `null`
(`"rope_scaling": null`). What actually runs is: standard Llama-style GQA
attention (48 query heads / 8 KV heads, `head_dim=128` — non-square, since
`hidden_size / num_heads = 3072/48 = 64 ≠ 128`), RoPE with an unusually large
`rope_theta = 70,000,000`, and a weight-shared loop over the 22 physical
decoder layers executed `num_loops=2` times (44 effective layers), with the
final RMSNorm re-applied at each loop boundary (`skip_loop_final_norm=false`).

This is corroborated independently by Nanbeige's own officially-documented
fast-path serving route: their `Nanbeige/ollama` fork (`nanbeige42` branch,
`x/models/nanbeige/nanbeige.go`) implements *only* plain GQA attention and a
trivial loop-repeat (`for il := 0; il < nLogical; il++ { phys := il % nPhys; ... }`)
— no mHC, no depth attention, no n-gram code at all. Worth naming plainly:
**the one official serving path that sidesteps the buggy generic-`transformers`
port entirely also never has to implement (or validate) any of the model's
more novel advertised capabilities**, because this checkpoint doesn't use
them. Anyone benchmarking or citing this release's "Looped Transformer +
mHC + depth attention + n-gram embeddings" architecture should know that, as
shipped, it's exercising exactly one of those four ideas.

## The five bugs

### Bug 1 — RoPE scaling dispatch (`KeyError: 'type'`)

`NanbeigeAttention._init_rope()` branches on `self.config.rope_scaling is None`,
then does `scaling_type = self.config.rope_scaling["type"]` in the `else`
branch, assuming any non-`None` `rope_scaling` has a `"type"` key. This
checkpoint's `config.json` sets `rope_scaling: null` — but something in the
installed `transformers` version populates `config.rope_scaling` with a dict
lacking a `"type"` key by the time `NanbeigeAttention` is instantiated
(`transformers` has been migrating RoPE-scaling config shape library-wide).
Result: `KeyError: 'type'` at model construction time, before any generation
is even attempted.

**Fix:** treat a `rope_scaling` dict without a usable `"type"` the same as "no
scaling" — matching what this checkpoint's own `config.json` actually asks
for. Independently validated against Nanbeige's own Go MLX reimplementation,
whose `RopeScaling` struct treats `"", "default"` as an identical no-op.

### Bug 2 — `Cache.get_max_length` → `get_max_cache_shape` sentinel mismatch

`transformers` renamed `DynamicCache.get_max_length` to `get_max_cache_shape`,
and changed what "no maximum" means: the old method returned `None` for a
dynamic/unbounded cache; the new one returns `-1` for the same case (per its
own docstring). `modeling_nanbeige.py` calls the old name/semantics in three
places (`_update_causal_mask`, twice in `prepare_inputs_for_generation`). A
naive `get_max_length = get_max_cache_shape` alias (the first fix attempted)
passes `-1` straight through instead of translating it back to `None`, which
makes a normally-dead code branch reachable — see Bug 2b.

**Fix:** a real compatibility helper, `_get_max_cache_length_compat()`, that
calls `get_max_cache_shape` when available and translates any negative
sentinel back to `None`.

### Bug 2b (consequence of a naive Bug 2 fix, not independent) — tuple/int `TypeError`

With the naive `-1`-passthrough alias, `prepare_inputs_for_generation`'s
`cache_length + input_ids.shape[1] > max_cache_length` check becomes
reachable and raises `TypeError: can only concatenate tuple (not "int") to
tuple`. Confirmed via the model's own official usage example, which never
passes an `attention_mask` to `generate()` — short-circuiting that whole `if`
block via Python's `and` operator, meaning the model's own maintainers never
exercised this path. Resolved by fixing Bug 2 properly rather than patching
this symptom directly.

### Bug 3 — `position_ids` not re-trimmed (the actual MPS crash)

The real cause of a hard, uncatchable native crash on Apple Silicon:
```
'mps.matmul' op contracting dimensions differ 361 & 181
MPSGraphExecutable.mm:1484: failed assertion 'original module failed verification'
```
`prepare_inputs_for_generation` retrieves a pre-supplied `position_ids` kwarg
via `kwargs.get("position_ids", None)`. The installed `transformers` version's
generation loop passes this on every decode step, covering the *full
accumulated sequence* (not just the new token(s)). The only re-trim logic
(`position_ids = position_ids[:, -input_ids.shape[1]:]`) lives inside
`if attention_mask is not None and position_ids is None:` — which never runs
once `position_ids` is already non-`None`. `input_length` is then computed
from the stale, oversized `position_ids` instead of the correctly-trimmed
`input_ids`, corrupting `cache_position` into a multi-element range instead of
the single new position — producing mismatched Q/K/V sequence lengths deep in
attention. Confirmed via direct instrumentation of `prepare_inputs_for_generation`
across real `generate()` calls: `position_ids` arrives as shape `(1, 181)` on
the decode step for token 182, and is never re-trimmed.

CUDA appears to tolerate or silently mishandle the same shape bug rather than
hard-crashing — MPS's compiled Metal graph backend does not, which is likely
why this was never caught on the hardware the model was developed against
(the official usage example uses `.to("cuda")`).

**Fix:** add exactly one branch — `elif position_ids is not None and
position_ids.shape[1] != input_ids.shape[1]: position_ids =
position_ids[:, -input_ids.shape[1]:]` — mirroring what the branch above
already does for itself. Verified by instrumenting every single decode step
of a full generation: `cache_position`/`position_ids` are length-1 and
increment by exactly 1 at every step, from token 1 through the end.

### Bug 4 — `inv_freq` silently zeroed (the dominant bug)

The one that actually explains the structural incoherence (character-level
word salad, looping `</think>` tokens) that persisted even after Bugs 1–3
were fixed and the crash was gone — regardless of dtype, sampling parameters,
`enable_thinking`, `max_new_tokens` budget, attention backend, or device
(reproduces identically on CPU/float32/eager, ruling out anything MPS- or
bf16-specific).

`NanbeigeRotaryEmbedding.__init__` computes `inv_freq` correctly, then
registers it via `register_buffer("inv_freq", inv_freq, persistent=False)`.
The installed `transformers` version loads weights through a meta-device-init
+ checkpoint-restore path: non-persistent buffers have no entry in the
checkpoint (that's the intent of `persistent=False` — they're meant to be
cheap to recompute), so nothing repopulates them once the model materializes
off the meta device. The `__init__`-computed values are simply lost.

Confirmed directly: every `inv_freq` value in a loaded model was exactly
`0.0`, all 64 entries, for every layer:
```
shape: torch.Size([64])
min/max: 0.0 0.0
num zeros: 64 / 64
```
It should range from `1.0` down to roughly `0.03` (for `head_dim=128`,
`rope_theta=70_000_000`):
```
expected first 10: [1.0, 0.7540850639343262, 0.5686442852020264,
0.4288061559200287, 0.32335636019706726, 0.24383819103240967,
0.1838747262954712, 0.13865718245506287, 0.10455932468175888,
0.07884661853313446]
```
A zero `inv_freq` means every position gets an identical, zero rotation:
**RoPE contributes no positional information at all.** The model is
structurally blind to word order and token position — which is exactly the
kind of severe, low-level breakdown that produces incoherent, repetitive
output regardless of any generation-time setting.

This was found by rereading a comment in a third-party MLX port's source
([`jishnuvenugopal/nanbeige-mlx`](https://github.com/jishnuvenugopal/nanbeige-mlx),
`nanbeige_mlx/model.py`), which noted in passing that the HF reference "holds
`inv_freq` as a non-persistent fp32 buffer" as a caveat about a different,
much smaller issue (a hypothetical dtype-casting bug that port avoids by
construction, since MLX's `nn.RoPE` stores four Python scalars instead of a
buffer). Rereading that comment against our own loaded model surfaced this
instead. That project is independent and unaffiliated with Nanbeige; its code
was reviewed by hand before trusting the lead, to the same standard as
reviewing Nanbeige's own custom code.

**Fix, two parts:**
1. `register_buffer("inv_freq", inv_freq, persistent=True)` — so the value is
   actually included in this checkpoint's `state_dict` going forward.
2. The checkpoint's weights were re-saved after loading once with `inv_freq`
   recomputed correctly in memory, so the *correct values* are baked directly
   into this repo's `model.safetensors` — not dependent on any future
   loader's meta-device/buffer-persistence behavior at all. Verified with a
   cold process, no runtime patches: `inv_freq[:5] = [1.0, 0.754..., 0.569...,
   0.429..., 0.323...]`.

### Bug 5 — `_tied_weights_keys` list vs. dict format

Unrelated to generation correctness, but breaks `save_pretrained()`:
`NanbeigeForCausalLM._tied_weights_keys = ["lm_head.weight"]` uses the old
list-of-names format. The installed `transformers`'s
`_get_tied_weight_keys()` now expects a dict mapping (tied name → source
name) and calls `.keys()` on it unconditionally:
`AttributeError: 'list' object has no attribute 'keys'`. This checkpoint sets
`tie_word_embeddings: false`, so `lm_head.weight` isn't actually tied to
anything here — the attribute only matters if a future Nanbeige checkpoint
enables weight tying.

**Fix:** `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`.

## Elimination methodology (for the output-quality investigation)

Before finding Bug 4, every other plausible explanation for the incoherent
output was tested and ruled out directly, not assumed:

- **Not sampling parameters** — reproduces under this project's hardcoded
  test settings, under plain greedy decoding, and under Nanbeige's own
  shipped `generation_config.json` defaults (`temperature=0.6, top_p=0.95,
  top_k=20`, no repetition penalty).
- **Not `enable_thinking`** — reproduces with it forced off, forced on, and
  left at the chat template's default.
- **Not token budget** — reproduces at 60, 100, 300, and 800 `max_new_tokens`.
- **Not `eos_token_id`** — confirmed correct (`166101`, `<|im_end|>`) at the
  tokenizer, model config, and generation config level.
- **Not the chat template** — `apply_chat_template` renders a well-formed,
  complete ChatML prompt in every configuration tested.
- **Not dtype, not MPS, not the attention backend** — reproduces identically
  on CPU, `float32`, and eager attention, ruling out any Apple
  Silicon/bf16/SDPA-specific explanation before Bug 4 was found.

## What's unaffected

- **Dtype**: this checkpoint's native `torch_dtype` is `bfloat16` (per its own
  `config.json`); the weights here were re-saved in `bfloat16` throughout.
  Loading in `float16` still works but loses precision given the unusually
  large `rope_theta` — use `bfloat16` if your hardware supports it.
- **`vllm`'s Transformers backend** (`--model-impl transformers`) hasn't been
  validated against this checkpoint specifically — vLLM's paged-attention KV
  cache management differs fundamentally from HF's `DynamicCache`/`generate()`
  loop that Bugs 2/2b/3 target; those specific fixes may be moot (harmless,
  just unused) under that serving path. Bugs 1, 4, and 5 all happen at model
  construction/loading time and should still apply regardless of serving
  stack.

## License

Code changes: MIT, matching the license of the community project whose
comment led to Bug 4. Model weights and the base custom code remain governed
by the upstream `Nanbeige/Nanbeige4.2-3B` Apache-2.0 license; see `LICENSE`.
Changes made from the base revision (`5d54321e9e01e0d026f8e371046678fc384dca39`)
are documented in full above, per Apache-2.0 §4.
