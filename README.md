# pointer-agent

A from-scratch demonstration of an **SLM (planner) → VLM (visual grounder) → SLM
(verifier)** pipeline for GUI pointer/click-target grounding — trained with SFT
and RLVR, benchmarked against [PointerBench](https://huggingface.co/datasets/WarmwindOS/pointerbench)
(`pointerbench-text` subset) and scored with the benchmark's own scorer.

Given a screenshot and a natural-language instruction (e.g. `Point to "cores".`
or `Draw a box around "city".`), the pipeline returns an absolute pixel
coordinate or bounding box for the target.

## Architecture

```
instruction ──► SLM planner ──► {anchor_phrase, relation,     ──► VLM grounder ──► anchor bbox
                (Qwen2.5)         relation_params, answer_type}   (Florence-2)          │
                                                                                        ▼
                                                        relation == "self"?  ──yes──►  done
                                                                 │no
                                                                 ▼
                                                  crop to anchor bbox (+margin)
                                                                 │
                                                                 ▼
                                                  BaseCharLocator (Tesseract OCR)
                                                                 │
                                                                 ▼
                                                  deterministic geometric resolution
                                                  (edges / char offsets / gaps —
                                                   plain Python, zero model calls)
                                                                 │
                                                                 ▼
                                                          SLM verifier ──► final prediction
```

Every component sits behind a small ABC (`orchestrator/base.py`) so a new
planner/grounder/verifier/locator plugs in via a Hydra config change, not an
orchestration-code change.

### Design note: anchor + relation + deterministic resolution (Sprint 4)

The planner's output shape is fixed for every category:
`{anchor_phrase, relation, relation_params, answer_type}`. The VLM
(Florence-2) only ever does one thing — locate `anchor_phrase` in the full
image via its native `<CAPTION_TO_PHRASE_GROUNDING>` head. `relation ==
"self"` (word/invoice/structural-text categories) means that grounded bbox
*is* the answer. Every other relation (caret/char/line-edge/paragraph-edge/
between-anchors — 11 in total) is resolved by a deterministic
`BaseCharLocator` (`orchestrator/char_locator.py`, concrete
`TesseractCharLocator`) from a crop around the grounded anchor: plain OCR +
geometry, zero model calls, zero training.

This replaced an earlier (Sprint 3) design where the VLM was asked to
directly ground relational phrases like "just before the word 'cores'" -
which turned out to be architecturally mismatched, since phrase-grounding
VLMs find objects/phrases but aren't trained to represent boundaries,
offsets, or ordinal character positions precisely. Full category→relation
mapping for all 39 synthetic categories, the bugs found while building this
(an invoice VLM-SFT phrase bug, a caret-family ground-truth precision gap, a
missing `data-text` DOM attribute, and the anchor-bbox-vs-final-bbox
distinction needed for correct VLM SFT training), and the
crop→full-image coordinate-transform correctness test are recorded in
`progress.md` sections 4b/4c.

## Repo layout

```
pointer-agent/
├── data_gen/            # synthetic training-data generator (Playwright + Jinja2 + Faker)
│   └── templates/        # HTML/CSS surface templates
├── orchestrator/         # SLM planner -> VLM grounder -> SLM verifier pipeline
├── slm/                  # planner LoRA-SFT (Phase 2): dataset + train_sft.py
├── vlm/                  # grounder LoRA-SFT (Phase 1): dataset + train_sft.py
├── eval/                 # wraps the real pointerbench-text scorer; runs it against any source
├── scripts/              # schema validation + ground-truth visualization tooling
├── configs/              # Hydra config groups (data_gen, model/*, train/*, hardware/*, eval)
│   └── accelerate/        # accelerate launcher configs (device placement, separate from Hydra)
├── output/               # generated data / eval / training runs (gitignored-scale artifacts)
├── requirements.txt
└── requirements-train.txt  # GPU-only extras (bitsandbytes) for the real SFT runs
```

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate

# torch/torchvision must be installed as a matched pair from the CPU wheel index.
# Installing torchvision from plain PyPI against a mismatched torch build breaks
# its compiled ops ("operator torchvision::nms does not exist").
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
playwright install chromium

# System binary required by orchestrator/char_locator.py (TesseractCharLocator) -
# pytesseract (in requirements.txt) is just a thin wrapper around this binary.
sudo apt-get install -y tesseract-ocr   # Debian/Ubuntu
# brew install tesseract                 # macOS

# Only needed to actually run the SFT training scripts on a real GPU
# (bitsandbytes doesn't build/run meaningfully on a CPU-only machine):
pip install -r requirements-train.txt
```

Notes on the pinned versions in `requirements.txt`:
- `transformers==4.49.0` is pinned **below 5.0** — Florence-2's `trust_remote_code`
  config class breaks on transformers 5.x (a `PretrainedConfig` attribute-access
  change). Confirmed working with both Qwen2.5-Instruct and Florence-2-base.
- No GPU is required to run anything in this repo today — all models used so far
  are CPU-sized. `device` is a config field throughout, so pointing at `cuda` on
  a Colab/GPU box is a config override, not a code change.
- `pytesseract` needs the system `tesseract-ocr` binary installed separately
  (not pip-installable) — see above.

## Usage

### 1. Generate synthetic training data

```bash
python -m data_gen.generate_dataset n_samples=200 output_dir=output/synthetic_v0
```

Hydra-driven (`configs/data_gen/config.yaml`) — override any field on the
command line, e.g. `languages='[en]'` or
`push_to_hub=true hub_repo_id=<you>/pointer-agent-synth`. Defaults to a local
dry-run save (`datasets.save_to_disk`, no HF token needed); set `push_to_hub=true`
+ `hub_repo_id`/`hub_token` to actually push once credentials are available.

Verify the output:

```bash
python scripts/validate_schema.py output/synthetic_v0
python scripts/visualize_samples.py output/synthetic_v0 --limit 10
```

`validate_schema.py` checks internal consistency (bbox validity, point-in-bbox,
required fields, image files present) and can diff field names against a real
`pointerbench-text metadata.jsonl` via `--real <path>`. `visualize_samples.py`
draws each row's bbox/point back onto its screenshot under `<output_dir>/viz/`.

### 2. Run the Phase 0 baseline eval

```bash
python -m eval.run_eval source=synthetic limit=30
python -m eval.run_eval source=real limit=30
```

Hydra-driven (`configs/eval/phase0.yaml`, composing the `model/planner`,
`model/grounder`, `model/verifier` config groups) — swap any component with an
override once an alternative config exists, e.g. `model/grounder=<other>`.
`source=real` downloads the real `pointerbench-text` metadata + only the
sampled images from HF Hub on demand (not the full ~500-image set).

Each run writes `predictions.jsonl`, `report.json`, and bbox/point overlay
images (predicted in blue, ground truth in red) under
`output/eval/phase0_<source>/`.

**Full-scale runs (all 500 real + 200+ synthetic rows) are a Colab/GPU job**,
not a local one — the current dev machine is CPU-only. Raise or drop `limit`
and point `device` at `cuda` to run there; no code changes needed.

### 3. LoRA-SFT the VLM / SLM (Phase 1 + Phase 2)

```bash
# Phase 1: grounder (Florence-2), LoRA on the language-model attention
# projections, native <loc_N> location-token targets (not JSON coordinates -
# see progress.md for why that generalizes better here).
python -m vlm.train_sft hardware=single_gpu_t4 data.metadata_path=output/synthetic_v1/metadata.jsonl

# Phase 2: planner (Qwen2.5), LoRA on q/k/v/o_proj, instruction -> structured-query pairs.
python -m slm.train_sft hardware=single_gpu_t4 data.metadata_path=output/synthetic_v1/metadata.jsonl
```

Both scripts are Hydra-driven (`configs/train/{vlm_sft,slm_sft}.yaml`) and
hardware-profile agnostic via `configs/hardware/{cpu_smoketest,single_gpu_t4,
multi_gpu}.yaml` (batch size, precision, gradient checkpointing, 4-bit
quantization). Training runs through a plain HF `Trainer` (already
accelerate-integrated), so multi-GPU/multi-node is a launcher flag, not a
code change:

```bash
accelerate launch --config_file configs/accelerate/multi_gpu.yaml -m vlm.train_sft hardware=multi_gpu
```

`hardware=cpu_smoketest` (the default) runs a few steps on a tiny subset to
prove the script is correct — it does not train a usable model. Each script
saves a LoRA adapter to `cfg.save_adapter_dir`; point
`configs/model/{grounder/florence2_base_sft,planner/qwen2_5_0_5b_sft}.yaml`'s
`adapter_path` at it (or override on the CLI) and re-run
`eval/run_eval.py model/grounder=florence2_base_sft` /
`model/planner=qwen2_5_0_5b_sft` for the Phase 1 / Phase 2 numbers.

## Status

### Implemented

**Sprint 1 — synthetic data generation** (`data_gen/`)
- Renders real HTML/CSS via headless Chromium (Playwright) and reads exact
  ground-truth pixel boxes from the DOM (`getBoundingClientRect()`) — no
  hand-drawn or OCR'd ground truth.
- 3 surface templates: `document`, `code_editor`, `server_inventory`.
- 4 task categories: `word_center`, `word_bbox` (direct-reference), `line_start`,
  `line_end` (positional, line-clustered by y-coordinate).
- 2 languages: `en`, `de`.
- Output schema matches the real `pointerbench-text` dataset field-for-field
  (`file_name`, `id`, `instruction`, `bbox`, `point`, `answer_type`, `eval`,
  `data_type`, `category`, `surface`, `language`, `difficulty`, `image_size`),
  so the same eval harness runs unmodified against synthetic data, PointerBench,
  or any future source.
- `push_to_hub.py` — local dry-run save by default, real HF Hub push behind a
  config flag.

**Sprint 2 — baseline MVP, end-to-end pipeline / Phase 0** (`orchestrator/`, `eval/`)
- Prompted-only SLM planner (`Qwen/Qwen2.5-0.5B-Instruct`) parses an instruction
  into `{target_phrase, answer_type}`.
- Off-the-shelf VLM grounder (`microsoft/Florence-2-base`) grounds the phrase
  via its native `<CAPTION_TO_PHRASE_GROUNDING>` task head — no fine-tuning.
- Text-only SLM verifier (shared Qwen weights with the planner): plain-Python
  geometry check + one LM call asking whether the matched phrase plausibly
  answers the instruction; triggers one grounder retry with a
  punctuation-stripped phrase on failure.
- `eval/scorer.py` imports the real `pointerbench-text` `eval.py` directly
  (`importlib`, downloaded from HF Hub) rather than reimplementing its
  point-in-bbox / coverage-precision rules.
- Verified end-to-end on a 50-row synthetic subset and a 30-row real
  `pointerbench-text` subset: **0.00% accuracy on both**. Root cause confirmed
  by inspecting predictions directly, not assumed: 44/50 (88%) synthetic
  predictions are near-whole-image fallback boxes (Florence-2 found no real
  match), and the remaining 6 still miss by a wide margin (e.g. predicting a
  ~900x360px region for a target that's actually 43x17px). This is a genuine
  off-the-shelf grounding failure on small UI/document text, not a pipeline
  bug. Zero `missing_predictions` in either report; this is the Phase 0
  baseline number Phase 1 (VLM SFT) needs to beat.

**Sprint 3 — SFT the VLM + SLM, data-gen hardening** (`data_gen/`, `slm/`, `vlm/`)
- `data_gen/` hardened to full spec breadth: 7 surfaces (added `invoice`,
  `magazine_spread`, `chat_ui`, `form`), 39 task categories across every
  family (word/character/caret/line/paragraph-level, relative, plus 19
  invoice field-extraction categories), all 6 languages (`en`, `de`, `fr`,
  `es`, `it`, `nl`), and curriculum knobs (distractors, occlusion overlays,
  broader font-scale/theme variation). Category sampling is uniform
  round-robin across all requested categories, per-row surface eligibility
  gated by category. Per-character DOM spans (needed for zero-noise
  char/caret/paragraph ground truth) are scoped to prose surfaces
  (`document`, `magazine_spread`) to bound render cost.
- `slm/` (new) and `vlm/` (new): LoRA-SFT training scripts for Phase 2
  (planner: instruction → structured query) and Phase 1 (grounder: phrase →
  Florence-2 native `<loc_N>` location tokens, not a JSON-coordinate output
  head — see `progress.md` for why that generalizes better). Both are
  Hydra-driven and hardware-profile agnostic (`configs/hardware/{cpu_smoketest,
  single_gpu_t4,multi_gpu}.yaml`), running through a plain accelerate-integrated
  HF `Trainer` so multi-GPU/multi-node is `accelerate launch`, not a code change.
- `orchestrator/` extended (backward-compatibly) with an optional
  `adapter_path` on `QwenSLM`/`Florence2Grounder` and an optional
  `referring_expression` field on the planner's query contract (used for
  caret/relative-position categories); Phase 0 configs are untouched.
- Verified via CPU smoke tests only (a handful of steps on tiny data, finite
  loss, adapter save/reload/inference proof) — the real full-scale training
  run is a Colab/GPU job left for the user, per the project's compute
  constraints (this dev machine is CPU-only).
- **Not done this sprint**: `BaseGrounder`/`BasePlanner` swap validated with a
  second concrete model (e.g. Qwen2-VL) — deferred, still open.

**Sprint 4 — architecture refinement** (`orchestrator/`, `data_gen/`) — see
the Architecture section above for the design
- Planner's output contract rewritten to `{anchor_phrase, relation,
  relation_params, answer_type}` (`orchestrator/planner_qwen.py`,
  `DEFAULT_SYSTEM_PROMPT` is the single source of truth for the JSON shape).
  `referring_expression` retired.
- New `BaseCharLocator` ABC + `TesseractCharLocator`
  (`orchestrator/char_locator.py`) resolving all 11 non-`self` relations
  from an OCR'd crop around the VLM-grounded anchor.
  `orchestrator/pipeline.py` branches on `relation` accordingly (`self` /
  char-locator / two-VLM-call `between_anchors` geometry).
- Three data-gen bugs found and fixed: invoice categories now ground on the
  semantic field label, not the literal (ungroundable-by-the-planner) value;
  `structural_text` instructions now quote the literal title text (its DOM
  container was also missing a `data-text` attribute entirely); VLM SFT
  training now targets the anchor's own box (`anchor_bbox`), not the task's
  final answer box, which differ for every relational category.
  `line_start`/`line_end`/`paragraph_start`/`paragraph_end` switched to
  genuine caret-width ground truth; char/caret/punctuation categories
  narrowed to `en`/`de` to match the real benchmark.
- `eval/run_eval.py` reports relation-classification accuracy alongside
  coordinate accuracy (only for `source=synthetic`).
- Verified: `tests/test_char_locator.py` (2/2 passed against the real
  `tesseract-ocr` binary — the required crop→full-image coordinate-transform
  proof); a full pipeline smoke test ran clean, no exceptions, across every
  relation type; 0% coordinate accuracy (expected — nothing retrained yet)
  with 75% relation-classification accuracy zero-shot from the prompted
  planner. See `progress.md` sections 4b/4c for the full design + build log.
- **Not done this sprint**: retraining `slm/train_sft.py`/`vlm/train_sft.py`
  against this new contract (the pre-Sprint-4 SFT runs targeted the retired
  schema) — that's a Colab/GPU job, tracked as a pending item below.

### Pending

**Retrain the SFT models against the Sprint 4 schema**
- `slm/train_sft.py` and `vlm/train_sft.py`'s CPU smoke tests (Sprint 3) ran
  against the pre-Sprint-4 `target_phrase`/`referring_expression` contract,
  which no longer exists. Both datasets/scripts already read the new
  `anchor_phrase`/`anchor_bbox`/`relation`/`relation_params` schema, but
  haven't been re-run (smoke or full-scale) since. Re-run the CPU smoke
  tests, then the real Colab/GPU training, then re-evaluate
  (`model/planner=qwen2_5_0_5b_sft`, `model/grounder=florence2_base_sft`) for
  real Phase 1/2 + relation-classification accuracy numbers.

**W&B experiment tracking** (spec requirement, not yet wired up)
- `slm/train_sft.py` and `vlm/train_sft.py` currently set `report_to=[]` —
  stdout logging only. The original spec calls for W&B throughout (SFT, RL,
  eval runs); this was deliberately kept out of Sprint 3 to keep it focused
  on data-gen breadth + SFT plumbing, and is tracked here instead of being
  silently dropped. Needs: `wandb` added to `requirements.txt`,
  `report_to=["wandb"]` + `wandb.init(...)` (config-gated so CPU smoke tests
  don't require a W&B account) in both train scripts, and equivalent logging
  in the eval harness and the future RL loop.

**Sprint 5 — joint RLVR loop (Phase 3)** (split out of the original Sprint 4,
once architecture refinement lands — needs its own algorithm/reward-shaping
decisions, so gets its own dedicated planning pass)
- `rl/`: `BaseReward` interface + RLVR reward function reusing the eval
  scorer's own hit logic.
- Shared-reward, separate-backward-pass update: one rollout reward, independent
  policy-gradient updates for SLM and VLM (not a fused graph) — keeps memory
  bounded on a single T4.
- Verifier wired into the RL loop as an active re-query trigger, not just a
  passive pass-through.
- Multi-GPU/multi-node Hydra hardware profile alongside the single-T4 one.

**Sprint 6 — optional RLHF, scale-out, polish** (was Sprint 5)
- Optional Phase 4: human preference pairs on ambiguous cases → small reward
  model → PPO against it.
- Validate the multi-GPU/multi-node profile actually runs.
- ScreenSpot / ScreenSpot-v2 as secondary generalization eval sets.
- `REPRODUCE.md`, pinned `requirements.txt`, `CITATION.cff` — mirroring
  PointerBench's own packaging standard.
- Full phase-by-phase (0→4) results table across both PointerBench and
  secondary eval sets.

**Sprint 7 — verifier-triggered perturbation retry (post-optimization
failsafe)** (was Sprint 6)
- On verification failure, re-run grounding against a slightly offset/
  perturbed crop instead of the same input, to escape a bad local grounding
  result. Deliberately sequenced after Sprint 4's architecture refinement and
  Sprint 5's RLVR loop are both working — a failsafe refinement, not a
  core-pipeline gap.

Full background, design rationale, and constraints are in the original project
spec (not checked into this repo as a file — see project conversation history).
