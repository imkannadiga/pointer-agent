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
instruction ──► SLM planner ──► structured query ──► VLM grounder ──► point/bbox
                (Qwen2.5)         {target_phrase,       (Florence-2)        │
                                   answer_type}                             ▼
                                                                    SLM verifier
                                                                    (shared Qwen)
                                                                     │ retry on fail
                                                                     ▼
                                                              final prediction
```

Every component sits behind a small ABC (`orchestrator/base.py`) so a new
planner/grounder/verifier plugs in via a Hydra config change, not an
orchestration-code change.

## Repo layout

```
pointer-agent/
├── data_gen/            # synthetic training-data generator (Playwright + Jinja2 + Faker)
│   └── templates/        # HTML/CSS surface templates
├── orchestrator/         # SLM planner -> VLM grounder -> SLM verifier pipeline
├── eval/                 # wraps the real pointerbench-text scorer; runs it against any source
├── scripts/              # schema validation + ground-truth visualization tooling
├── configs/              # Hydra config groups (data_gen, model/*, eval)
├── output/               # generated data / eval runs (gitignored-scale artifacts)
└── requirements.txt
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
```

Notes on the pinned versions in `requirements.txt`:
- `transformers==4.49.0` is pinned **below 5.0** — Florence-2's `trust_remote_code`
  config class breaks on transformers 5.x (a `PretrainedConfig` attribute-access
  change). Confirmed working with both Qwen2.5-Instruct and Florence-2-base.
- No GPU is required to run anything in this repo today — all models used so far
  are CPU-sized. `device` is a config field throughout, so pointing at `cuda` on
  a Colab/GPU box is a config override, not a code change.

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
- Verified end-to-end on 30-row subsets of both synthetic and real data:
  **0.00% accuracy on both**, confirmed (by inspecting raw Florence-2 output)
  to be a genuine off-the-shelf grounding failure — it returns near-whole-image
  boxes for small UI/document text — not a pipeline bug. Zero
  `missing_predictions` in both reports; this is the Phase 0 baseline number
  Phase 1 (VLM SFT) needs to beat.

### Pending

**Sprint 3 — SFT the VLM + SLM, data-gen hardening (Phase 1 + Phase 2)**
- Broaden `data_gen/` to full spec breadth: remaining surface templates
  (`magazine_spread`, `chat_ui`, `form`, full invoice-field template), the
  remaining task families (relative, attribute-based, caret/character-level,
  invoice extraction), all 6 languages (`en`, `de`, `fr`, `es`, `it`, `nl`), and
  the difficulty/curriculum knobs (distractors, occlusion, font-scale, theme noise).
- LoRA-SFT the VLM on synthetic data (SLM stays frozen/prompted); re-eval to
  isolate VLM-attributable gains.
- LoRA-SFT the SLM planner on instruction→structured-query pairs; re-eval to
  isolate SLM-attributable gains.
- `BaseGrounder`/`BasePlanner` swap validated with a second concrete model
  (e.g. Qwen2-VL as an alternative grounder).

**Sprint 4 — joint RLVR loop (Phase 3)**
- `rl/`: `BaseReward` interface + RLVR reward function reusing the eval
  scorer's own hit logic.
- Shared-reward, separate-backward-pass update: one rollout reward, independent
  policy-gradient updates for SLM and VLM (not a fused graph) — keeps memory
  bounded on a single T4.
- Verifier wired into the RL loop as an active re-query trigger, not just a
  passive pass-through.
- Multi-GPU/multi-node Hydra hardware profile alongside the single-T4 one.

**Sprint 5 — optional RLHF, scale-out, polish**
- Optional Phase 4: human preference pairs on ambiguous cases → small reward
  model → PPO against it.
- Validate the multi-GPU/multi-node profile actually runs.
- ScreenSpot / ScreenSpot-v2 as secondary generalization eval sets.
- `REPRODUCE.md`, pinned `requirements.txt`, `CITATION.cff` — mirroring
  PointerBench's own packaging standard.
- Full phase-by-phase (0→4) results table across both PointerBench and
  secondary eval sets.

Full background, design rationale, and constraints are in the original project
spec (not checked into this repo as a file — see project conversation history).
