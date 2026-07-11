"""LoRA capacity sanity-check: can the VLM overfit a tiny fixed subset?

Trains the exact same setup as vlm/train_sft.py (same dataset class, same
collate, same LoRA targets, same task prompt) on a handful of rows for many
epochs, then greedy-decodes those *same* rows through the same
post-processing the real grounder uses (orchestrator/grounder_florence2.py).

Interpretation:
- Loss collapsing toward ~0 AND the trained boxes being reproduced at
  inference (high IoU / hit rate vs. the near-zero pre-training baseline)
  => the LoRA has the capacity to learn this problem; scaling data/steps is
  what stands between you and accuracy.
- Loss plateauing high or inference staying at whole-image fallbacks even
  on memorized samples => capacity/wiring problem (rank, target modules,
  quantization, label masking), which more data will NOT fix.

Usage:
    # GPU box (IAS-1275: GPU 1 is faulty - pin to GPU 0):
    CUDA_VISIBLE_DEVICES=0 python -m vlm.overfit_test

    # More pressure / different subset:
    CUDA_VISIBLE_DEVICES=0 python -m vlm.overfit_test --epochs 100 --lr 1e-4

    # CPU works too (slow - expect ~30 min at the defaults):
    python -m vlm.overfit_test --device cpu --epochs 20
"""
import argparse

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor

from orchestrator.grounder_florence2 import TASK_PROMPT
from vlm.dataset import GroundingSFTDataset, make_collate_fn

MODEL_NAME = "microsoft/Florence-2-base"
# Mirrors configs/train/vlm_sft.yaml's lora block.
LORA = dict(r=816, alpha=32, dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])


def iou(a: list, b: list) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter) if inter else 0.0


@torch.no_grad()
def evaluate(model, processor, dataset, device, label: str, show: int = 5) -> dict:
    """Greedy-decode every sample through the real grounder decode path and
    score the predicted bbox against the trained target (anchor_bbox)."""
    model.eval()
    ious, hits = [], 0
    print(f"\n--- {label} ---")
    for i in range(len(dataset)):
        row = dataset.rows[i]
        sample = dataset[i]
        inputs = processor(text=sample["prompt"], images=sample["image"],
                           return_tensors="pt").to(device)
        out = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256, num_beams=1, do_sample=False, early_stopping=False,
        )
        text = processor.batch_decode(out, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            text, task=TASK_PROMPT,
            image_size=(sample["image"].width, sample["image"].height),
        )
        bboxes = parsed.get(TASK_PROMPT, {}).get("bboxes", [])
        pred = [round(v) for v in bboxes[0]] if bboxes else \
            [0, 0, sample["image"].width, sample["image"].height]  # grounder's fallback

        gt = row["anchor_bbox"]
        sample_iou = iou(pred, gt)
        cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
        hit = gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]
        ious.append(sample_iou)
        hits += hit
        if i < show:
            print(f"  [{row['id']}] '{row['anchor_phrase'][:40]}'  "
                  f"gt={gt}  pred={pred}  iou={sample_iou:.2f}  hit={hit}")
    mean_iou = sum(ious) / len(ious)
    print(f"  => mean IoU {mean_iou:.3f} | center-hit {hits}/{len(ious)} "
          f"| whole-image fallbacks {sum(1 for v in ious if v < 0.01 )} low-IoU(<0.01)")
    return {"mean_iou": mean_iou, "hit_rate": hits / len(ious)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metadata", default="output/synthetic_v1/metadata.jsonl")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Overfit tests conventionally run hotter than the "
                         "real run's 3e-5; both should memorize, 1e-4 just "
                         "gets there in fewer epochs.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
        attn_implementation="eager", torch_dtype=torch.float32,
    )
    model = get_peft_model(model, LoraConfig(
        r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"],
        target_modules=LORA["target_modules"], task_type=None,
    ))
    model.print_trainable_parameters()
    model.to(args.device)

    dataset = GroundingSFTDataset(args.metadata, max_samples=args.n_samples)
    print(f"Overfitting {len(dataset)} samples from {args.metadata} "
          f"for {args.epochs} epochs on {args.device}")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=make_collate_fn(processor),
        generator=torch.Generator().manual_seed(0),
    )

    before = evaluate(model, processor, dataset, args.device, "BEFORE training (baseline)")

    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    first_epoch_loss = None
    model.train()
    for epoch in range(1, args.epochs + 1):
        total, n = 0.0, 0
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # match Trainer default
            optim.step()
            optim.zero_grad()
            total, n = total + loss.item(), n + 1
        epoch_loss = total / n
        if first_epoch_loss is None:
            first_epoch_loss = epoch_loss
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}/{args.epochs}  loss {epoch_loss:.4f}")
        last_epoch_loss = epoch_loss

    after = evaluate(model, processor, dataset, args.device,
                     "AFTER training (same samples - memorization check)", show=10)

    print("\n================ VERDICT ================")
    print(f"loss: {first_epoch_loss:.4f} -> {last_epoch_loss:.4f}")
    print(f"mean IoU:   {before['mean_iou']:.3f} -> {after['mean_iou']:.3f}")
    print(f"center-hit: {before['hit_rate']:.0%} -> {after['hit_rate']:.0%}")
    if after["mean_iou"] >= 0.5 and after["hit_rate"] >= 0.8:
        print("PASS: the LoRA memorized the subset - capacity is not the bottleneck.")
    elif last_epoch_loss < first_epoch_loss * 0.3:
        print("PARTIAL: loss collapsed but decoded boxes are off - suspect the "
              "decode/quantization path or too few epochs, not raw capacity.")
    else:
        print("FAIL: cannot overfit 20 samples - capacity/wiring problem "
              "(try higher r/alpha, check label masking, LR).")


if __name__ == "__main__":
    main()
