# pointer-agent

A from-scratch demonstration of an **SLM (planner) → VLM (visual grounder) → SLM
(verifier)** pipeline for GUI pointer/click-target grounding — trained with SFT
and RLVR, benchmarked against [PointerBench](https://huggingface.co/datasets/WarmwindOS/pointerbench)
(`pointerbench-text` subset) and scored with the benchmark's own scorer.

Given a screenshot and a natural-language instruction (e.g. `Point to "cores".`
or `Draw a box around "city".`), the pipeline returns an absolute pixel
coordinate or bounding box for the target.

