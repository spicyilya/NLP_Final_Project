"""Fine-tuning machinery for the MELD emotion-classification grid.

Lives in src/ rather than inside notebook 03 so that the grid can be driven
either from the notebook or headless (`python src/training.py`), and so that a
crash or an interrupted session never loses completed runs: every finished run
is appended to results/hparam_search.csv immediately, and run_grid() skips any
configuration already present there.
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

try:
    from . import utils  # noqa: F401
except ImportError:  # executed as a script / from a notebook
    import utils

SEARCH_CSV = utils.RESULTS / "hparam_search.csv"

# Held fixed across the grid (stated in the report):
#   warmup_ratio 0.06 — the RoBERTa paper's value; a short warmup stabilizes the
#     first few hundred steps when the freshly-initialized classifier head emits
#     large gradients into the pretrained encoder.
#   weight_decay 0.01 — standard BERT fine-tuning default; mild regularization on
#     a ~10k-example corpus.
# Both are held constant so the grid isolates the effect of lr and batch size.
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 5


class WeightedTrainer(Trainer):
    """Trainer with a class-weighted cross-entropy loss.

    MELD is ~17.6x imbalanced; unweighted training collapses toward the neutral
    majority class. Weights come from notebook 02 (inverse train frequency).
    """

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = (
            self.class_weights.to(logits.device, dtype=logits.dtype)
            if self.class_weights is not None
            else None
        )
        loss = nn.CrossEntropyLoss(weight=weight)(
            logits.view(-1, model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def load_splits(names=("train", "dev")):
    return {n: pd.read_csv(utils.DATA_PROCESSED / f"{n}.csv") for n in names}


def make_dataset(df: pd.DataFrame, tokenizer):
    """Tokenize a split into a torch-ready HF Dataset (no padding — dynamic later)."""
    from datasets import Dataset

    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    ds = ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=utils.MAX_LENGTH),
        batched=True,
        remove_columns=["text"],
        desc="tokenizing",
    )
    return ds


def _completed_runs() -> set[tuple[str, float, int]]:
    if not SEARCH_CSV.exists():
        return set()
    done = pd.read_csv(SEARCH_CSV)
    return {(r.model, float(r.learning_rate), int(r.batch_size)) for r in done.itertuples()}


def _append_row(row: dict) -> None:
    """Append one finished run immediately — the CSV is the crash-safe log."""
    df = pd.DataFrame([row])
    header = not SEARCH_CSV.exists()
    df.to_csv(SEARCH_CSV, mode="a", header=header, index=False)


def train_one(
    model_key: str,
    lr: float,
    batch_size: int,
    class_weights: torch.Tensor,
    datasets: dict,
    tokenizer,
    out_root: Path,
) -> dict:
    """Fine-tune one configuration for up to MAX_EPOCHS, keeping the best dev epoch."""
    model_name = utils.MODELS[model_key]
    run_name = f"{model_key}_lr{lr:g}_bs{batch_size}"
    run_dir = out_root / run_name

    utils.set_seed(utils.SEED)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=utils.NUM_LABELS,
        id2label=utils.ID2LABEL,
        label2id=utils.LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(run_dir),
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=MAX_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_strategy="epoch",
        seed=utils.SEED,
        data_seed=utils.SEED,
        report_to="none",
        disable_tqdm=True,
        dataloader_num_workers=0,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=utils.compute_metrics,
        class_weights=class_weights,
    )

    t0 = time.time()
    trainer.train()
    wall = time.time() - t0

    # Best epoch = the one Trainer restored via load_best_model_at_end.
    evals = [h for h in trainer.state.log_history if "eval_weighted_f1" in h]
    best = max(evals, key=lambda h: h["eval_weighted_f1"])

    row = {
        "model": model_key,
        "model_name": model_name,
        "learning_rate": lr,
        "batch_size": batch_size,
        "max_epochs": MAX_EPOCHS,
        "best_epoch": int(best["epoch"]),
        "dev_weighted_f1": round(best["eval_weighted_f1"], 4),
        "dev_macro_f1": round(best["eval_macro_f1"], 4),
        "dev_accuracy": round(best["eval_accuracy"], 4),
        "dev_loss": round(best["eval_loss"], 4),
        "wall_time_s": round(wall, 1),
        "epoch_f1_trace": ";".join(f"{h['eval_weighted_f1']:.4f}" for h in evals),
    }

    # Keep this run's best checkpoint under a stable path; prune the rest.
    keep = run_dir / "best"
    if keep.exists():
        shutil.rmtree(keep)
    trainer.save_model(str(keep))
    tokenizer.save_pretrained(str(keep))
    for ckpt in run_dir.glob("checkpoint-*"):
        shutil.rmtree(ckpt, ignore_errors=True)

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    return row


def run_grid(
    model_keys=("bert", "roberta"),
    learning_rates=(1e-5, 2e-5, 5e-5),
    batch_sizes=(16, 32),
) -> pd.DataFrame:
    """Run the full grid, skipping configurations already logged in the CSV."""
    utils.ensure_dirs()
    class_weights = torch.tensor(utils.load_class_weights())
    splits = load_splits(("train", "dev"))
    done = _completed_runs()

    total = len(model_keys) * len(learning_rates) * len(batch_sizes)
    print(f"grid: {total} runs, {len(done)} already complete")

    for model_key in model_keys:
        tokenizer = AutoTokenizer.from_pretrained(utils.MODELS[model_key])
        datasets = {n: make_dataset(df, tokenizer) for n, df in splits.items()}
        out_root = utils.CHECKPOINTS / model_key

        for lr in learning_rates:
            for bs in batch_sizes:
                if (model_key, float(lr), int(bs)) in done:
                    print(f"skip  {model_key} lr={lr:g} bs={bs} (already logged)")
                    continue
                print(f"\n=== {model_key} lr={lr:g} bs={bs} ===", flush=True)
                row = train_one(model_key, lr, bs, class_weights, datasets, tokenizer, out_root)
                _append_row(row)
                print(
                    f"--> dev weighted F1 {row['dev_weighted_f1']:.4f} "
                    f"macro {row['dev_macro_f1']:.4f} "
                    f"(best epoch {row['best_epoch']}, {row['wall_time_s']:.0f}s)",
                    flush=True,
                )

        del datasets
        gc.collect()
        torch.cuda.empty_cache()

    return pd.read_csv(SEARCH_CSV)


def promote_best() -> dict:
    """Keep only the winning checkpoint per model; delete the rest. Returns best configs."""
    results = pd.read_csv(SEARCH_CSV)
    best_configs = {}
    for model_key, grp in results.groupby("model"):
        best = grp.loc[grp["dev_weighted_f1"].idxmax()]
        src = utils.CHECKPOINTS / model_key / f"{model_key}_lr{best.learning_rate:g}_bs{int(best.batch_size)}" / "best"
        dst = utils.CHECKPOINTS / model_key / "best"
        if src.resolve() != dst.resolve():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        for run_dir in (utils.CHECKPOINTS / model_key).glob(f"{model_key}_lr*"):
            shutil.rmtree(run_dir, ignore_errors=True)
        best_configs[model_key] = {
            "learning_rate": float(best.learning_rate),
            "batch_size": int(best.batch_size),
            "best_epoch": int(best.best_epoch),
            "dev_weighted_f1": float(best.dev_weighted_f1),
            "dev_macro_f1": float(best.dev_macro_f1),
            "checkpoint": str(dst.relative_to(utils.ROOT)),
        }
    with open(utils.RESULTS / "best_configs.json", "w", encoding="utf-8") as fh:
        json.dump(best_configs, fh, indent=2)
    return best_configs


if __name__ == "__main__":
    df = run_grid()
    print("\n" + df.to_string(index=False))
