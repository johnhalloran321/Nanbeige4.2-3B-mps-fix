---
title: "Running Nanbeige4.2-3B on Apple Silicon: everything that broke, and what we did about it"
layout: default
---

# Running Nanbeige4.2-3B on Apple Silicon: everything that broke, and what we did about it

[Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) is a 3B-parameter agentic
model that reports competitive results against much larger models on tool-use benchmarks,
credited to a "Looped Transformer" design that reuses one stack of layers twice per
token instead of stacking twice as many unique layers. That's an appealing shape for a
local, on-device agent backend: small on disk, allegedly strong for its size. We tried to
actually run it, as a real backend behind a real MCP-connected agent loop, on Apple
Silicon. It didn't work out of the box, and once we got it loading, it didn't work well.

This is the long-form version of the paper (arXiv link coming soon) — same numbers, same conclusions,
but without cramming five distinct findings into two sentences. If you want the
compressed, citable version, read the paper. If you want to understand *why* each of
these things happened, read on.

**TL;DR:** five independent bugs block the checkpoint from loading or running correctly
via Hugging Face `transformers` at all. Fixing them isn't enough — the architecture's own
memory and speed characteristics cause a real production out-of-memory crash and a severe
agentic-throughput problem that no amount of bug-fixing touches. We fix what's fixable
(chunked prefill for the memory side, a chat-template splice for a separate tool-calling
regression), measure what isn't, and are honest in the paper about which is which.

## Why we went looking

Small agentic models are attractive precisely because they're small: no API costs, no
data leaving the machine, no dependency on a serving stack that might not even support
the architecture. Nanbeige4.2-3B looked like a good test case — 3B non-embedding
parameters, competitive with Qwen3.5-4B and Qwen3.5-9B on agentic benchmarks. Running it
in a real ReAct-style agent loop, though, surfaced enough correctness and stability
problems that it was unusable out of the box: wrong output, unbounded memory growth, and
a chat-template bug that corrupted tool-calling the moment an agent framework supplied
its own system prompt. None of this is visible from the model card. None of it is exotic,
either — it's the ordinary, tedious kind of integration bug that blocks real deployment
even when a model evaluates well in isolation.

## Bug hunt: five ways it breaks before you even get to inference

We loaded the unmodified checkpoint via
`transformers.AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` and ran
it on an MPS device. Five independent bugs showed up, each confirmed by direct
reproduction against the unmodified checkpoint (exact line numbers and error strings are
in the paper's Table 1, against `transformers==5.8.1`).

### 1. The RoPE buffer is silently zeroed (the dominant bug)

This is the one that actually mattered most, and it's also the sneakiest, because it
never raises an exception. The model's `inv_freq` rotary-embedding buffer — the thing
that encodes *where* a token is in the sequence — gets zeroed on load and never
repopulated before the first forward pass. With `inv_freq` at zero, RoPE contributes
**zero positional information** to attention. The model runs. It generates fluent-looking
text. It just has no idea what order the tokens came in. You only catch this by directly
inspecting buffer values after load; nothing about the failure mode announces itself.

### 2. A RoPE-config dispatch `KeyError`

This is the first bug you actually hit, because it fires during model construction,
before device placement or a single forward pass — it blocks loading on *any* device, not
just MPS. A configuration-routing bug in the custom modeling code's RoPE-type dispatch
raises `KeyError: 'type'` for a subset of otherwise-valid config values.

### 3. A cache API that no longer exists

The custom `modeling_nanbeige.py` calls `DynamicCache.from_legacy_cache(...)`, an API
that's been removed from current `transformers` releases. We found this one independently
of the original investigation, while instrumenting the prefill-memory experiments below —
calling the model's `forward()` directly with `past_key_values=None` (bypassing
`generate()`) hits it immediately. Worth being honest about: our shipped patch doesn't
cover this path. It only fixes the case where a cache object is already explicit, which is
how our own harness always calls the model — so this bug is real, reproducible today, and
still there.

### 4. A position-IDs bug that only crashes on MPS

A position-ID re-trimming step in the custom attention code produces a hard,
uncatchable crash specifically on Apple Silicon's MPS backend. We could not reproduce it
identically on CPU or CUDA — which is exactly why it likely went unnoticed by whoever
developed and tested this model against CUDA hardware.

### 5. `save_pretrained()` is broken too

An incompatible tied-weights key naming convention breaks `save_pretrained()`, so even
after you've patched everything above and have a correctly-running model in memory, you
can't re-serialize it without an additional fix.

We fixed all five via sibling-file monkeypatching — never touching cached `transformers`
package files or the committed model code — and shipped the result as
[`johnhalloran/Nanbeige4.2-3B-mps-fix`](https://huggingface.co/johnhalloran/Nanbeige4.2-3B-mps-fix)
on Hugging Face.

## The memory tradeoff nobody mentions in the model card

Here's the part that isn't a bug, exactly — it's an architectural consequence that only
becomes a problem once you try to actually deploy the thing.

A Looped Transformer gets its parameter efficiency by running the *same* stack of
physical layers more than once. Nanbeige4.2-3B has 22 physical decoder layers, run
through twice (`num_loops=2`), for 44 effective layer-executions from 22 layers' worth of
weights. That's a genuine win for the parameter/quality tradeoff — you get more effective
depth without more disk space or more unique weights to store.

It is *not* a memory-savings story at inference time, though, and the model card doesn't
say so. Self-attention's peak activation memory during prefill is dominated by an
attention-score tensor proportional to `prompt_len²`, computed once per layer pass. A
looped model doesn't reduce that cost — it *doubles* it relative to a non-looped model
with the same physical layer count, because the identical quadratic-cost attention
computation over the same prompt runs twice. On a discrete GPU with dozens of gigabytes of
dedicated VRAM, that doubling is usually absorbable. On Apple Silicon's unified memory —
shared with the OS and every other process, with none of CUDA's page-out flexibility — it
isn't.

We know this isn't hypothetical, because it already happened: a single real production
request, with roughly 20+ MCP tool schemas serialized into its prompt, attempted a
**35.89 GiB** allocation on a machine with 34.36 GB of total unified memory. Immediate,
unrecoverable OOM.

### Reproducing the exact crash, and deriving a real number for "how long is too long"

We didn't want to estimate this — we wanted to reproduce it exactly. Using the same 11
connected MCP servers elsewhere in this project (55 tool schemas total, matching the
"20+" from the original incident), rendered through the real chat template and tokenized
with the real Nanbeige tokenizer, we get a prompt that is **exactly 12,244 tokens**. We
call this length *M* — it's not an estimate, it's a measured value, and it's the largest
length in every sweep we ran.

Naive prefill on this exact prompt doesn't just OOM the way the original crash did — it
hits an even harder failure mode. On Apple Silicon, a Metal-level assertion aborts the
whole process outright (`SIGABRT`), not a Python exception you could catch and retry.
Chunked prefill on the identical prompt doesn't hard-crash, but it still fails, with a
catchable `RuntimeError: MPS backend out of memory` — and it fails the *same way, every
time*, byte-identically across independent fresh-process runs. That reproducibility
matters: it means this is a deterministic property of the checkpoint at this length on
this hardware, not noise you might get lucky and avoid on a good day.

## Chunked prefill: a real fix, but a partial one

The actual fix is small: instead of a single `model(input_ids=full_prompt, ...)` call
that materializes the whole `prompt_len × prompt_len` attention-score tensor at once, we
process the prompt in fixed 256-token chunks, growing a `DynamicCache` incrementally
between them the same way normal autoregressive decoding already does. This bounds the
per-step attention tensor to `chunk_size × running_total` instead of
`prompt_len × prompt_len`, independent of how long the prompt actually is. We verified
bit-identical output against plain `generate()` on prompts short enough for naive prefill
to succeed, so this isn't a behavior change, just a memory-shape change.

We measured single-request and batched behavior at eight lengths (1024 up through the
full *M* = 12,244), using 50 real, individually-long documents from LongBench-Pro rather
than concatenated shorter texts — every measurement is on genuine, unmodified real-world
content, truncated (never padded or concatenated) to the target length.

Two things came out of this, and they don't point the same direction, so we report both
rather than picking the flattering one:

**Chunked prefill supports a meaningfully larger batch size** at every length where naive
prefill works at all — 2× at 1024 tokens, 4× at 2048, 2× at 4096 — and at 8192 tokens
naive can't complete even a single request while chunked handles a batch of one cleanly
and reproducibly.

**But at each method's own max batch size, naive is actually faster** in raw tokens/sec,
everywhere both produce a number. The reason is unglamorous: our chunked-prefill
implementation processes each sequence in the batch through several sequential sub-calls,
and that per-chunk overhead is paid per batch element instead of being amortized across
the whole batch the way a real batched-attention serving kernel (the kind vLLM implements)
would do. A bigger safe batch size is still genuinely useful — you can serve more
concurrent requests — but it doesn't translate into higher throughput without pairing
chunking with that kind of kernel, which our reference harness (built to demonstrate the
correctness/memory fix, not to be a production serving engine) doesn't have.

And critically: chunking extends the usable range, it doesn't close the gap. Naive fails
via an uncatchable abort by roughly 8,192–9,000 tokens even at batch size one. Chunked
prefill's real single-request ceiling sits between 11,231 and the full *M* = 12,244 — both
methods fail at the original production scale. Chunking buys you real headroom, not a
complete fix.

## The other half of the problem: decode is slow, and getting slower fixes prefill doesn't touch

Everything above is about *prefill* — processing the prompt before generation starts.
Autoregressive decode is a completely separate cost, and it turns out to be the one that
actually dominates real agentic use. We measured it directly: forced-length generation
(`min_new_tokens = max_new_tokens`, so early stopping can't bias the number) after chunked
prefill to a given context length.

| Context tokens | Decode tok/s |
|---|---|
| 64 | 15.5 |
| 1024 | 7.8 |
| 5120 | 2.1 |

Throughput falls by more than 7× between a trivial prompt and 5,120 tokens of context.
This is the same architectural cost as the memory story above, just showing up on the
other side of generation: every decode step also passes through both loop iterations of
the full layer stack, so decode cost per token grows with context the same way prefill's
does, stacked on top of the ordinary KV-cache attention cost every decoder-only model
already pays. As you'll see below, this — not prefill, not a correctness bug — is what
actually limits Nanbeige's usability as a real MCP-connected agent.

## A completely separate bug: the chat template silently discards its own system prompt

This one has nothing to do with memory. Nanbeige4.2-3B's chat template does a plain
if/else on the first message: if the caller supplies *any* system message, it's used
verbatim, with a trailing double-newline the template appends. If the caller supplies
nothing, the template injects the model's own hardcoded, trained-in tool-use system
prompt, with *no* trailing separator before the tools section that follows.

The problem: that's a *replace*, not a *merge*. Any caller-supplied system message —
exactly what a general-purpose agent harness like ironclad-agent always supplies —
silently discards the model's own trained default instead of extending it. We confirmed
this directly: the identical tools-plus-user-message request produces a clean,
correctly-formatted single tool call with no caller system message, and garbled,
malformed multi-tool-call syntax the instant *any* caller system message is added — even
one as anodyne as "You are a helpful assistant with MCP tools."

The fix isn't as simple as "just re-supply the default text yourself," either. We tried
that — re-supplying the extracted default verbatim through the explicit-system-message
branch still breaks, because that branch's own auto-appended whitespace differs from the
zero-extra-whitespace auto-insert branch's output by exactly **two characters**. This
model's tool-calling reliability is calibrated to the *exact* byte sequence its own
default rendering path produces, which makes sense if its tool-use training data was only
ever rendered through that one path and never with an explicit caller system message.

The actual fix: render the chat template with no system message at all, so it takes its
own zero-extra-whitespace auto-insert path, then splice the caller's system content
directly into the rendered string immediately after the auto-inserted default — never
through the template's asymmetric explicit-system-message branch.

## Evaluation: MCPMark, and why the headline number needs six footnotes

We evaluated the combined fixes against [MCPMark](https://arxiv.org/abs/2509.24002)'s
Filesystem task subset — real MCP servers, programmatically verified (no LLM-judge
scoring bias), no external credentials required. This section is the one where "a wall of
text" does the most damage, because there isn't one finding here — there are at least six,
and they matter in a specific order.

### The headline number, and why it's a trap

The unpatched checkpoint can't even be instantiated under a current `transformers`
release, so it scores a hard 0/10 by construction — not a comparable per-task score, just
a wall. The *patched* checkpoint, run against the same 10 tasks under MCPMark's own
default 600-second per-task budget, scores **1/10**. Five of six task categories show
average task time pinned exactly at the 600-second ceiling, with correspondingly tiny
turn counts (as low as zero turns for one category). Our first instinct was that this
looked like prefill latency at long context. We were wrong, and we think it's worth
saying so directly rather than quietly fixing the narrative — the trap here is that "0/10"
and "1/10" both *look* like a broken model, but the actual explanation took real digging.

### Ruling out prefill

We reconstructed the exact prompt MCPMark sent at the moment two representative tasks
stopped progressing — one that failed, one that succeeded — using the real 14-tool
filesystem schema and the real tokenizer. Both land in the same narrow band: 5,122 and
5,445 tokens. Chunked prefill finishes either in under 30 seconds. Prompt length does not
distinguish the failing task from the succeeding one. Whatever MCPMark is actually
exposing, it isn't prefill.

### The real cause: decode throughput

We replayed the stalled task's exact conversation state end to end, letting generation run
far longer than MCPMark's own budget allows. What actually happens: chunked prefill
finishes in 24.5 seconds, and decode then generates 471 tokens at **1.89 tokens/sec**
before stopping on its own. It is not stuck. It is not looping. It is simply far slower
than a 600-second budget assumes. The generated text is coherent the whole way through —
the model narrates an extended, genuinely thoughtful, repeatedly self-correcting
deliberation ("Let me count the characters," "Actually, let me," "Let me get the file size
first") before finally emitting a tool call. This lines up almost exactly with our
controlled decode measurement above (2.10 tok/s at ~5,100 tokens of context vs. 1.89
tok/s in the real replay). MCPMark also requests up to 32,768 tokens per turn, uncapped by
our harness, so a task that needs several such turns — each costing on the order of a
minute once context reaches a few thousand tokens — exhausts a 600-second budget through
ordinary accumulation, not any single catastrophic call.

### Giving it more time

We re-ran the same 10 tasks with the per-task timeout raised from 600 to 3,600
seconds — MCPMark's own tool default, not a number we picked to flatter the result.
Doing this surfaced a *second*, completely independent bug (see the next section), but
once that was fixed, the score moved from 1/10 to **3/10**. Two additional tasks crossed
the finish line, and a third made substantial visible progress (2 turns → 7 turns, 5.2K →
36K tokens of accumulated context) without quite finishing. More time recovers real,
additional capability — which is exactly what you'd expect from a throughput bottleneck,
and not what you'd expect from a correctness bug.

### An unplanned detour: a second memory bug

Raising the timeout didn't just give the model more time — it also gave a genuinely new
bug room to appear. A single caught `RuntimeError: MPS backend out of memory` turns out to
*permanently* degrade the harness process's usable memory budget for the rest of its
life. We tested this directly: neither `torch.mps.empty_cache()` nor `gc.collect()`
reclaim the lost memory — driver-allocated memory stayed pinned at 33–43 GB against an
~8 GB baseline even after explicit cleanup. Only a full process restart fixes it.
Uncorrected, one task's OOM (itself expected, since conversations can legitimately grow
past our measured memory ceiling) cascaded into spurious OOMs on every later, unrelated
task in the same long-lived server. The fix mirrors what we already did for the batch-size
sweep: restart the harness fresh before every task, so one task's failure can't poison the
next one's measurement.

### Is it a reasoning loop? No — and we checked directly rather than assuming

Five of the ten tasks stall at just two turns even under the full 3,600-second budget —
too few turns for "it's just slow" to be the whole story on its own, and exactly the kind
of pattern that *would* worry us if it turned out to be the model looping. So we checked,
directly: we replayed each stalled conversation against the model with the HTTP layer
removed (so any crash surfaces a full traceback) and generation allowed to run up to 8,000
tokens. None of the five produced repetitive, garbled, or non-terminating output. Every
single one terminated normally or hit a clean, deterministic error. Two distinct
mechanisms explain all five, and neither one is a reasoning loop.

### Mechanism one: the model is verbose in exactly the wrong way

In one task, the model correctly decided to read all 21 files in the task directory with
a single `read_multiple_files` call — genuinely reasonable tool use. But the call's JSON
arguments repeat a long absolute path 21 times, so generating this *one, valid,
well-formed* tool call costs 1,779 tokens and 723 seconds at 2.46 tok/s. Nothing is wrong
with the model's decision here. It's just that at this decode speed, a verbose but correct
tool call is itself expensive.

### Mechanism two: a single tool response can blow the memory budget by itself

In the other four stalled tasks, the model calls `directory_tree` or `search_files`
exactly once, as you'd expect — and the *tool's own response*, not anything the model
generated, comes back at 14,500 to 42,400 tokens. The MCP filesystem reference server's
`directory_tree` has no depth limit, size cap, or pagination. One fixture
(`student_database`) genuinely contains 451 files across 150 student folders, and the tool
dumps the entire pretty-printed tree in one response. That single tool result exceeds our
measured 12,244-token chunked-prefill ceiling by up to 3.5×, so the very next prefill
attempt OOMs — deterministically, every time. This is real, generalizable friction between
an unbounded, size-agnostic MCP tool and a memory-constrained model. Any model with a
comparable memory ceiling hits the same wall on a realistically-populated directory; it's
not specific to MCPMark.

### How much of this is MCPMark's own fault?

A fair question, and one we tested rather than argued about. MCPMark's own
backup-directory naming scheme adds a real prefix to every path in every tool call or
response — in our environment, about 190 characters, before you even get to the actual
filename. We relocated a copy of MCPMark to a short root (cutting that shared prefix to 86
characters) and re-ran three representative tasks under otherwise identical conditions.
The result lines up exactly with the two mechanisms above: the task whose bloat came from
the *model* repeating the path, and the task whose bloat came from a tool that echoes full
paths per match, both shrank substantially (47% and 41% respectively) — with the
`search_files` task moving from an immediate crash to a slower, non-crashing timeout
instead. The task whose bloat came from `directory_tree`'s bare filenames was completely
unaffected (a 0.3% difference, consistent with noise). Shortening the deployment path is a
real, free win for tools that echo full paths — but it does nothing for the more
fundamental issue: an unbounded directory listing over a genuinely large real directory
will still exceed this model's memory ceiling no matter where that directory lives on
disk.

## Evaluation: BFCL, or what happens when you remove the clock

MCPMark's score is dominated by decode throughput and tool-response size — not by whether
the model calls tools *correctly*. To isolate correctness on its own, we separately
evaluated the patched checkpoint against the
[Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)'s
non-live, single-turn categories: one correct call, picking 1 of N candidate functions,
emitting 2+ calls to the same function, emitting 2+ calls to different functions, and
correctly declining to call anything at all. Every one of these completes in a single
10–20 second generation, so there's no wall-clock or multi-turn-accumulation confound at
all — this measures format correctness, full stop.

| Category | Score | Dominant failure mode |
|---|---|---|
| Correctly decline an irrelevant call | 100% | — |
| One correct call | 63.3% | wrong call count |
| Pick 1 of N functions | 43.3% | wrong call count |
| 2+ calls, different functions | 30.0% | wrong number of functions |
| 2+ calls, same function | 3.3% | wrong number of functions |

The model is genuinely good at knowing when *not* to call a tool, and reasonably good at
a single well-specified call. But it's specifically, consistently weak at emitting
*multiple* tool calls in one turn — almost every failure in both parallel-call categories
is the model producing one call where two were required. That's a distinct, format-level
limitation. It would show up even on hardware fast enough to make decode throughput a
complete non-issue, and it's worth reporting on its own rather than folding it into the
same "it's just slow" story as MCPMark.

## Try it yourself

Everything above has a script behind it. See the repo's [README](../README.md) for setup,
or jump straight to:

- [`patch/`](../patch/) — the five bug fixes, as a diff against the original model code
- [`harness/`](../harness/) — the minimal server we ran every measurement against
- [`experiments/`](../experiments/) — every script referenced above, one per finding
- [`results/`](../results/) — the raw JSON and MCPMark transcripts behind every number
- the paper (arXiv link coming soon) — the same content, compressed for citation

The patched checkpoint is on Hugging Face as
[`johnhalloran/Nanbeige4.2-3B-mps-fix`](https://huggingface.co/johnhalloran/Nanbeige4.2-3B-mps-fix).
