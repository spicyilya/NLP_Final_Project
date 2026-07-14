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
# # 04 — Evaluation & comparison
#
# Loads the best checkpoint per model and evaluates both on the **test** split.
#
# Covers assignment **steps 8 and 9**.
#
# > **This is the first and only use of the test split.** Notebooks 01–03 use
# > train and dev exclusively; every modelling decision (max_length, class
# > weights, learning rate, batch size, epoch count) was made on dev. Test is
# > read once, here, to report final numbers.

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path.cwd().parent / "src"))
import utils  # noqa: E402

utils.ensure_dirs()
utils.set_seed()
sns.set_theme(style="whitegrid", context="notebook")

test = pd.read_csv(utils.DATA_PROCESSED / "test.csv")
search = pd.read_csv(utils.RESULTS / "hparam_search.csv")
with open(utils.RESULTS / "best_configs.json", encoding="utf-8") as fh:
    best_configs = json.load(fh)

print(f"test utterances: {len(test)}")
best_configs

# %% [markdown]
# ## 1. Predict on test

# %%
@torch.no_grad()
def predict(model_key: str, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Return logits for `texts` from the promoted best checkpoint of `model_key`."""
    ckpt = utils.CHECKPOINTS / model_key / "best"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).cuda().eval()

    out = []
    for i in range(0, len(texts), batch_size):
        batch = tok(
            texts[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=utils.MAX_LENGTH,
            return_tensors="pt",
        ).to("cuda")
        out.append(model(**batch).logits.float().cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


y_true = test["label"].to_numpy()
texts = test["text"].tolist()

logits, preds = {}, {}
for model_key in utils.MODELS:
    logits[model_key] = predict(model_key, texts)
    preds[model_key] = logits[model_key].argmax(-1)
    print(f"{model_key}: predicted {len(preds[model_key])} utterances")

# %% [markdown]
# ## 2. Headline metrics (step 8)
#
# A **majority-class baseline** (always predict neutral) is included as the
# reference point — on a 48%-neutral test split, accuracy alone is easy to
# mistake for competence.

# %%
rows = []

majority = np.zeros_like(y_true)  # neutral == label id 0
rows.append({
    "model": "majority baseline (always neutral)",
    "weighted_f1": f1_score(y_true, majority, average="weighted", zero_division=0),
    "macro_f1": f1_score(y_true, majority, average="macro", zero_division=0),
    "accuracy": accuracy_score(y_true, majority),
})

for model_key, p in preds.items():
    cfg = best_configs[model_key]
    rows.append({
        "model": utils.MODELS[model_key],
        "weighted_f1": f1_score(y_true, p, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, p, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, p),
        "lr": cfg["learning_rate"],
        "batch_size": cfg["batch_size"],
        "best_epoch": cfg["best_epoch"],
        "dev_weighted_f1": cfg["dev_weighted_f1"],
    })

test_metrics = pd.DataFrame(rows)
# Round only the metric columns — rounding `lr` would flatten 2e-5 to 0.0.
metric_cols = ["weighted_f1", "macro_f1", "accuracy", "dev_weighted_f1"]
test_metrics[metric_cols] = test_metrics[metric_cols].round(4)
test_metrics.to_csv(utils.RESULTS / "test_metrics.csv", index=False)
test_metrics

# %% [markdown]
# ## 3. Per-class breakdown

# %%
per_class = []
for model_key, p in preds.items():
    rep = classification_report(
        y_true, p, target_names=utils.LABEL_NAMES, output_dict=True, zero_division=0
    )
    for name in utils.LABEL_NAMES:
        per_class.append({
            "model": model_key,
            "emotion": name,
            "precision": round(rep[name]["precision"], 4),
            "recall": round(rep[name]["recall"], 4),
            "f1": round(rep[name]["f1-score"], 4),
            "support": int(rep[name]["support"]),
        })

per_class_df = pd.DataFrame(per_class)
per_class_df.to_csv(utils.RESULTS / "test_per_class.csv", index=False)
per_class_df.pivot(index="emotion", columns="model", values="f1").loc[utils.LABEL_NAMES]

# %%
for model_key, p in preds.items():
    print(f"\n===== {utils.MODELS[model_key]} =====")
    print(classification_report(y_true, p, target_names=utils.LABEL_NAMES, zero_division=0, digits=3))

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
plot_df = per_class_df.copy()
plot_df["emotion"] = pd.Categorical(plot_df["emotion"], categories=utils.LABEL_NAMES, ordered=True)
sns.barplot(data=plot_df, x="emotion", y="f1", hue="model", ax=ax, palette="colorblind")
ax.set_title("Per-class F1 on test (ordered by train frequency, most → least common)")
ax.set_ylabel("F1")
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=7, padding=1)
fig.tight_layout()
fig.savefig(utils.FIGURES / "per_class_f1.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Confusion matrices (step 8)
#
# Row-normalized: each row sums to 1, so cell (i, j) reads "of all true class
# i utterances, what fraction were predicted j". Normalizing by row is what
# makes rare classes legible — in raw counts, fear (50 test utterances) is
# invisible next to neutral (1,256).

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (model_key, p) in zip(axes, preds.items()):
    cm = confusion_matrix(y_true, p, normalize="true")
    disp = ConfusionMatrixDisplay(cm, display_labels=utils.LABEL_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False, xticks_rotation=45)
    ax.set_title(f"{utils.MODELS[model_key]}\n(row-normalized)")
fig.tight_layout()
fig.savefig(utils.FIGURES / "confusion_matrices.png", dpi=150)
plt.show()

# %%
# Top confusion pairs (off-diagonal), by rate and by raw count.
conf_rows = []
for model_key, p in preds.items():
    cm_rate = confusion_matrix(y_true, p, normalize="true")
    cm_count = confusion_matrix(y_true, p)
    for i, true_name in enumerate(utils.LABEL_NAMES):
        for j, pred_name in enumerate(utils.LABEL_NAMES):
            if i != j:
                conf_rows.append({
                    "model": model_key,
                    "true": true_name,
                    "predicted": pred_name,
                    "rate": round(cm_rate[i, j], 4),
                    "count": int(cm_count[i, j]),
                })

confusions = pd.DataFrame(conf_rows)
confusions.to_csv(utils.RESULTS / "confusion_pairs.csv", index=False)

for model_key in preds:
    top = confusions[confusions.model == model_key].nlargest(3, "rate")
    print(f"\n{utils.MODELS[model_key]} — top 3 confusion pairs by rate:")
    for _, r in top.iterrows():
        print(f"  {r['true']:9s} -> {r['predicted']:9s}  {100 * r['rate']:5.1f}% ({r['count']} utterances)")

# %%
# How much of the total error is "collapsed into neutral"?
for model_key, p in preds.items():
    wrong = y_true != p
    into_neutral = wrong & (p == utils.LABEL2ID["neutral"])
    from_neutral = wrong & (y_true == utils.LABEL2ID["neutral"])
    print(f"{model_key:8s} errors={wrong.sum():4d} | "
          f"predicted neutral when it wasn't: {into_neutral.sum():4d} ({100*into_neutral.sum()/wrong.sum():.1f}% of errors) | "
          f"true neutral misread as emotion: {from_neutral.sum():4d} ({100*from_neutral.sum()/wrong.sum():.1f}%)")

# %% [markdown]
# ## 5. Did class weighting do what we hoped? (step 4c revisited)
#
# The weighted loss should trade precision for recall on rare classes. If it
# worked, rare classes should show recall clearly above what their frequency
# would predict, with precision below their recall.

# %%
weighting_check = per_class_df.merge(
    pd.DataFrame({
        "emotion": utils.LABEL_NAMES,
        "train_pct": [100 * c / 9989 for c in [4710, 1743, 1205, 1109, 683, 271, 268]],
    }),
    on="emotion",
)
weighting_check["recall_minus_precision"] = (
    weighting_check["recall"] - weighting_check["precision"]
).round(4)
weighting_check.sort_values(["model", "train_pct"], ascending=[True, False])[
    ["model", "emotion", "train_pct", "precision", "recall", "recall_minus_precision"]
]

# %% [markdown]
# ## 6. Error analysis (step 9)
#
# The sample below is drawn from the errors the better model made with **high
# confidence** — cases where it was not merely uncertain but confidently wrong.

# %%
best_model = max(best_configs, key=lambda k: best_configs[k]["dev_weighted_f1"])
print(f"error analysis on the better model by dev F1: {utils.MODELS[best_model]}\n")

probs = torch.softmax(torch.tensor(logits[best_model]), dim=-1).numpy()
p = preds[best_model]
err = pd.DataFrame({
    "text": test["text"],
    "gold": [utils.ID2LABEL[i] for i in y_true],
    "pred": [utils.ID2LABEL[i] for i in p],
    "confidence": probs.max(-1).round(3),
    "n_words": test["text"].str.split().str.len(),
})
err = err[err.gold != err.pred]

sample = err.nlargest(10, "confidence")
sample.to_csv(utils.RESULTS / "error_examples.csv", index=False)
sample

# %% [markdown]
# ### Are short utterances harder? No — and this refutes the obvious hypothesis
#
# The intuitive story is that short, generic lines ("What?", "Hey.") should be
# *hardest*, because their emotion lives in delivery and dialogue context that a
# text-only model cannot see. We tested that directly. **It is false**: short
# utterances are markedly *easier*.
#
# Two candidate explanations are ruled out below:
#
# 1. *"Short utterances are more often neutral, and the model leans neutral."*
#    No — the neutral share is nearly identical in both groups (49.0% vs 47.8%).
# 2. *"It's a class-mix artifact."* No — short utterances score higher on
#    **both** neutral recall and non-neutral recall, and on macro F1, which is
#    insensitive to class mix.
#
# The likelier explanation is that short emotional lines in this corpus are
# **formulaic and lexically explicit** ("Oh my God!", "I'm so sorry") — the
# emotion word *is* the utterance. Longer turns carry mixed or hedged content
# where the emotional cue is diluted across a sentence.
#
# This does not rescue the context limitation — §5's ambiguity check shows that
# is real — but it does relocate it. The problem is not utterance *length*.

# %%
correct_len = test.loc[y_true == p, "text"].str.split().str.len()
error_len = test.loc[y_true != p, "text"].str.split().str.len()
print(f"median words — correct: {correct_len.median():.0f} | errors: {error_len.median():.0f}")

short = (test["text"].str.split().str.len() <= 3).to_numpy()
for name, mask in [("<= 3 words", short), ("  > 3 words", ~short)]:
    is_neutral = y_true[mask] == utils.LABEL2ID["neutral"]
    print(
        f"\n{name} (n={mask.sum():4d}): "
        f"acc={accuracy_score(y_true[mask], p[mask]):.3f}  "
        f"weighted_f1={f1_score(y_true[mask], p[mask], average='weighted', zero_division=0):.3f}  "
        f"macro_f1={f1_score(y_true[mask], p[mask], average='macro', zero_division=0):.3f}"
    )
    print(f"    true neutral: {100 * is_neutral.mean():.1f}%  |  "
          f"neutral recall: {(p[mask][is_neutral] == utils.LABEL2ID['neutral']).mean():.3f}  |  "
          f"non-neutral recall: {(p[mask][~is_neutral] == y_true[mask][~is_neutral]).mean():.3f}")

print("\n=> Short utterances are easier on every axis, including macro F1 —"
      "\n   so this is not explained by class mix.")

# %% [markdown]
# ### The real ceiling: identical text, different gold label
#
# This is the honest, quantitative version of the "no dialogue context" claim.
# The same string appears in test with **different gold labels** — an
# irreducible ceiling for any text-only model, since identical input cannot
# produce two different outputs. Whatever distinguishes an angry "Hey!" from a
# joyful one is simply not in the text.

# %%
dup = test.groupby("text")["label_name"].nunique()
ambiguous = dup[dup > 1]
n_ambig_utts = test["text"].isin(ambiguous.index).sum()
print(f"distinct strings with >1 gold label in test: {len(ambiguous)}")
print(f"test utterances affected: {n_ambig_utts} ({100 * n_ambig_utts / len(test):.1f}%)")
print("\nexamples:")
for text in ambiguous.head(5).index:
    labels = sorted(test.loc[test.text == text, "label_name"].unique())
    print(f"  {text!r:40s} -> {labels}")

# %% [markdown]
# ## Summary
#
# Written up in full in `report.md` §7–8. Artifacts produced here:
#
# - `results/test_metrics.csv` — headline comparison + majority baseline
# - `results/test_per_class.csv` — per-class precision/recall/F1
# - `results/confusion_pairs.csv` — every off-diagonal confusion
# - `results/error_examples.csv` — high-confidence errors
# - `results/figures/confusion_matrices.png`, `per_class_f1.png`
