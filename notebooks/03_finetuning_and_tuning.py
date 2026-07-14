# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python (nlp2 venv)
#     language: python
#     name: nlp2-venv
# ---

# %% [markdown]
# # 03 — Fine-tuning & hyperparameter search
#
# Fine-tunes `bert-base-uncased` and `roberta-base` on MELD under an identical
# protocol and runs a 12-run hyperparameter grid.
#
# Covers assignment **steps 5, 6, 7**.
#
# The training machinery lives in `src/training.py` rather than in this
# notebook, for two reasons: the grid is hours of GPU work and must survive an
# interrupted session, and the same code should be runnable headless
# (`python src/training.py`). Every finished run is appended to
# `results/hparam_search.csv` immediately, and `run_grid()` skips any
# configuration already logged — so **this notebook is idempotent**: re-running
# it re-renders the analysis without repeating completed GPU work.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

sys.path.insert(0, str(Path.cwd().parent / "src"))
import training  # noqa: E402
import utils  # noqa: E402

utils.ensure_dirs()
utils.set_seed()
sns.set_theme(style="whitegrid", context="notebook")

print(f"torch {torch.__version__} | CUDA {torch.version.cuda}")
print(f"device: {torch.cuda.get_device_name(0)}")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

# %% [markdown]
# ## 1. Training setup (steps 5–6)
#
# **Full fine-tuning** — every parameter of the encoder is updated, plus a
# freshly initialized 7-way classification head over the pooled `[CLS]`/`<s>`
# representation.
#
# The loss is a **class-weighted cross-entropy** (`WeightedTrainer` in
# `src/training.py`), using the inverse-frequency weights computed on train in
# notebook 02. Without it, the 17.6× imbalance pushes the model toward
# answering "neutral" to everything.
#
# Both models get an **identical protocol** — same splits, same weights, same
# grid, same seed, same selection rule — so any difference between them is
# attributable to pretraining rather than to unequal tuning effort.

# %%
print("class weights (by label id):")
for name, w in zip(utils.LABEL_NAMES, utils.load_class_weights()):
    print(f"  {name:9s} {w:.3f}")

print(f"\nfixed across the grid: warmup_ratio={training.WARMUP_RATIO}, "
      f"weight_decay={training.WEIGHT_DECAY}, max_epochs={training.MAX_EPOCHS}")
print(f"max_length={utils.MAX_LENGTH} (dynamic padding via DataCollatorWithPadding)")

# %% [markdown]
# ## 2. The grid (step 7)
#
# | Hyperparameter | Values | Justification |
# |---|---|---|
# | learning rate | 1e-5, 2e-5, 5e-5 | spans the range recommended by the BERT paper for fine-tuning; above ~5e-5 fine-tuning tends to diverge, below ~1e-5 it underfits within a few epochs |
# | batch size | 16, 32 | BERT-paper defaults; both fit 12 GB at seq len 128 in bf16. Crossed with lr rather than tuned separately because the two interact (larger batches average gradients over more examples, tolerating larger steps) |
# | epochs | ≤5, best epoch chosen on dev | MELD is small (~10k), so overfitting typically appears by epoch 3–4. Selecting the best epoch on dev makes epochs a *tuned* quantity without spending a separate run per value |
#
# 3 learning rates × 2 batch sizes = **6 runs per model, 12 total**.
#
# `warmup_ratio=0.06` and `weight_decay=0.01` are **held fixed** so the grid
# isolates the lr × batch-size interaction. Warmup stabilizes the first few
# hundred steps, when the randomly initialized head sends large gradients back
# into the pretrained encoder; the decay value is the standard BERT default.
#
# **Selection uses dev only — the test split is not touched in this notebook.**

# %%
# Resumable: skips configurations already present in results/hparam_search.csv.
results = training.run_grid()
results.sort_values("dev_weighted_f1", ascending=False)

# %% [markdown]
# ## 3. Search results

# %%
table = results[
    ["model", "learning_rate", "batch_size", "best_epoch",
     "dev_weighted_f1", "dev_macro_f1", "dev_accuracy", "wall_time_s"]
].sort_values(["model", "learning_rate", "batch_size"])
table

# %%
# Dev weighted F1 as a function of lr, one line per (model, batch size).
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, metric in zip(axes, ["dev_weighted_f1", "dev_macro_f1"]):
    sns.lineplot(
        data=results, x="learning_rate", y=metric, hue="model", style="batch_size",
        markers=True, dashes=True, ax=ax, palette="colorblind", markersize=9,
    )
    ax.set_xscale("log")
    ax.set_title(metric)
    ax.set_xlabel("learning rate (log scale)")
fig.suptitle("Hyperparameter search — dev performance")
fig.tight_layout()
fig.savefig(utils.FIGURES / "hparam_search.png", dpi=150)
plt.show()

# %% [markdown]
# ### Which hyperparameter mattered more?
#
# Spread of dev weighted F1 attributable to each knob: the range across
# learning rates (holding model/batch fixed) versus across batch sizes
# (holding model/lr fixed).

# %%
lr_spread = results.groupby(["model", "batch_size"])["dev_weighted_f1"].agg(lambda s: s.max() - s.min())
bs_spread = results.groupby(["model", "learning_rate"])["dev_weighted_f1"].agg(lambda s: s.max() - s.min())
print("dev weighted-F1 range across LEARNING RATES (per model+bs):")
print(lr_spread.round(4).to_string())
print(f"  mean spread: {lr_spread.mean():.4f}")
print("\ndev weighted-F1 range across BATCH SIZES (per model+lr):")
print(bs_spread.round(4).to_string())
print(f"  mean spread: {bs_spread.mean():.4f}")

# %%
# Per-epoch dev F1 traces — shows where overfitting sets in.
fig, ax = plt.subplots(figsize=(9, 4.5))
for _, r in results.iterrows():
    trace = [float(x) for x in str(r["epoch_f1_trace"]).split(";")]
    ax.plot(range(1, len(trace) + 1), trace, marker="o", alpha=0.75,
            label=f"{r['model']} lr={r['learning_rate']:g} bs={int(r['batch_size'])}")
ax.set_xlabel("epoch")
ax.set_ylabel("dev weighted F1")
ax.set_title("Dev weighted F1 per epoch (all runs)")
ax.set_xticks(range(1, training.MAX_EPOCHS + 1))
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(utils.FIGURES / "epoch_traces.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Best configuration per model
#
# The winning checkpoint per model is promoted to `checkpoints/{model}/best`
# and the losing runs' checkpoints are deleted (they are large and
# regenerable). Notebook 04 loads exactly these two.

# %%
best_configs = training.promote_best()
for model_key, cfg in best_configs.items():
    print(f"{model_key:8s} lr={cfg['learning_rate']:g} bs={cfg['batch_size']} "
          f"epoch={cfg['best_epoch']} -> dev weighted F1 {cfg['dev_weighted_f1']:.4f} "
          f"macro {cfg['dev_macro_f1']:.4f}")
best_configs

# %%
results.to_csv(utils.RESULTS / "hparam_search.csv", index=False)
print("search table + best_configs.json written to results/")

# %% [markdown]
# ## Summary
#
# - 12 runs (2 models × 3 learning rates × 2 batch sizes), ≤5 epochs each, with
#   the best epoch selected on dev weighted F1.
# - Full search table logged to `results/hparam_search.csv` — a required
#   deliverable for step 7.
# - Best checkpoint per model promoted to `checkpoints/{model}/best`.
# - Test split still untouched.
#
# Next: `04_evaluation_comparison.py`.
