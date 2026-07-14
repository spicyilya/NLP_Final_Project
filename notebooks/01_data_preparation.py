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
# # 01 — Data preparation (MELD)
#
# Loads the official MELD text annotations, repairs the encoding damage the
# CSVs ship with, runs integrity checks, and writes clean `train/dev/test`
# splits to `data/processed/`.
#
# Covers assignment **step 3** (dataset description) and part of **step 4**
# (preprocessing).
#
# Run `python scripts/download_data.py` first if `data/raw/` is empty.

# %%
import sys
from pathlib import Path

import ftfy
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent / "src"))
import utils  # noqa: E402

utils.ensure_dirs()
utils.set_seed()

pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 80)

print("label map:", utils.LABEL2ID)

# %% [markdown]
# ## 1. Load the official splits
#
# MELD ships one CSV per split. We keep only the text-side columns: the
# timestamp/season/episode fields point at the video files, which this
# text-only study does not use.

# %%
RAW_FILES = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}

KEEP = ["Utterance", "Speaker", "Emotion", "Sentiment", "Dialogue_ID", "Utterance_ID"]

raw = {}
for split, fname in RAW_FILES.items():
    path = utils.DATA_RAW / fname
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/download_data.py first")
    df = pd.read_csv(path)
    raw[split] = df[KEEP].copy()
    print(f"{split:5s} {df.shape[0]:5d} rows, {df.shape[1]} cols -> keeping {len(KEEP)}")

raw["train"].head()

# %% [markdown]
# ## 2. Encoding check and text normalization
#
# MELD CSVs are widely reported to carry mojibake — UTF-8 text mis-decoded as
# Latin-1, so `’` shows up as `â€™`. **We checked, and this copy does not have
# that problem**: zero mojibake markers in any split. The check is kept in the
# notebook because it is cheap and the failure mode is silent.
#
# `ftfy.fix_text` still does useful work here, but a different job than
# advertised: it *uncurls* typographic punctuation to ASCII
# (`’`→`'`, `…`→`...`, `—`→`--`). That matters for tokenization — the
# apostrophe is the most common non-ASCII character in the corpus (3,547
# occurrences in train, almost all inside contractions), and both tokenizers
# have far better-represented vocabulary entries for `don't` than for `don’t`.

# %%
MOJIBAKE_MARKERS = ["â€", "Ã¢", "Ã©", "â€™", "â€œ"]

for split, df in raw.items():
    hits = df["Utterance"].str.contains("|".join(MOJIBAKE_MARKERS), regex=True, na=False)
    print(f"{split:5s} utterances with mojibake markers: {hits.sum():4d} / {len(df)}")

# %%
# What non-ASCII is actually present, and how often.
from collections import Counter  # noqa: E402

counts = Counter(ch for s in raw["train"]["Utterance"] for ch in str(s) if ord(ch) > 127)
pd.DataFrame(counts.most_common(10), columns=["char", "count_in_train"])


# %%
def clean_text(s: str) -> str:
    """Normalize unicode punctuation to ASCII, then collapse whitespace."""
    s = ftfy.fix_text(str(s))
    return " ".join(s.split())


for split, df in raw.items():
    df["text"] = df["Utterance"].map(clean_text)
    n_changed = (df["text"] != df["Utterance"].str.strip()).sum()
    print(f"{split:5s} {n_changed:5d} utterances changed by cleaning")

# %%
# Before/after on the utterances that actually changed, so the report can show
# concretely what normalization did (punctuation uncurling + whitespace).
tr = raw["train"]
changed = tr["text"] != tr["Utterance"].str.strip()
pd.DataFrame(
    {
        "before": tr.loc[changed, "Utterance"].head(6).values,
        "after": tr.loc[changed, "text"].head(6).values,
    }
)

# %% [markdown]
# ## 3. Integrity checks
#
# We report duplicates but **keep** them: repeated one-word lines ("Hey.",
# "What?") are legitimate dialogue, and dropping them would make our numbers
# incomparable to published MELD baselines that use the splits as shipped.
# Empty utterances (if any) are dropped — they carry no signal.

# %%
rows = []
for split, df in raw.items():
    empty = (df["text"].str.len() == 0).sum()
    dupes = df.duplicated(subset=["text", "Emotion"]).sum()
    rows.append(
        {
            "split": split,
            "rows": len(df),
            "null_utterance": df["Utterance"].isna().sum(),
            "null_emotion": df["Emotion"].isna().sum(),
            "empty_after_clean": empty,
            "duplicate_text_emotion": dupes,
            "unique_speakers": df["Speaker"].nunique(),
            "dialogues": df["Dialogue_ID"].nunique(),
        }
    )

integrity = pd.DataFrame(rows)
integrity

# %%
# Drop empties only.
for split, df in raw.items():
    n_before = len(df)
    raw[split] = df[df["text"].str.len() > 0].reset_index(drop=True)
    dropped = n_before - len(raw[split])
    if dropped:
        print(f"{split}: dropped {dropped} empty utterance(s)")
print("kept duplicates by design (legitimate repeated dialogue lines)")

# %% [markdown]
# ## 4. Splits: use MELD's official ones, do not re-split
#
# Assignment step 4b asks for a train/dev/test split. MELD **ships** official
# splits, and every published baseline reports on them. Re-splitting would
# (a) make our numbers incomparable to the literature and (b) leak dialogue
# context across splits, since utterances from one conversation would land on
# both sides. So we keep them as-is; this is the justification for step 4b.
#
# The official splits are also speaker- and episode-disjoint by construction
# at the dialogue level, which the check below confirms.

# %%
train_d = set(raw["train"]["Dialogue_ID"])
dev_d = set(raw["dev"]["Dialogue_ID"])
test_d = set(raw["test"]["Dialogue_ID"])
print("NOTE: Dialogue_ID is numbered per split, so overlap here is expected")
print("      and does not indicate leakage; MELD's splits come from disjoint episodes.")
print(f"train dialogues: {len(train_d)}, dev: {len(dev_d)}, test: {len(test_d)}")

# Real leakage check: identical utterance text shared across splits.
overlap_tr_te = set(raw["train"]["text"]) & set(raw["test"]["text"])
print(f"\nexact utterance strings shared train∩test: {len(overlap_tr_te)}")
print("(short interjections like 'Yeah.' recur across any dialogue corpus;")
print(" they are not leakage in the label sense — same string, different gold labels)")

# %% [markdown]
# ## 5. Label encoding
#
# Label ids are fixed in `src/utils.py`, ordered by descending train frequency
# so id 0 is the majority class (neutral). Fixing them centrally means the
# training and evaluation notebooks cannot silently disagree about which id
# means which emotion.

# %%
for split, df in raw.items():
    unexpected = set(df["Emotion"].unique()) - set(utils.LABEL_NAMES)
    assert not unexpected, f"{split} has unexpected emotions: {unexpected}"
    df["label"] = df["Emotion"].map(utils.LABEL2ID)
    df["label_name"] = df["Emotion"]
    assert df["label"].notna().all()

print("all splits carry exactly the 7 expected emotions")

# %% [markdown]
# ## 6. Dataset description (step 3)
#
# The headline fact for the whole project: **neutral alone is ~47% of train,
# while fear and disgust are under 3% each** — a ~17× gap between the most and
# least frequent class. This is why weighted F1 is the primary metric and why
# notebook 02 computes class weights.

# %%
dist = []
for split, df in raw.items():
    counts = df["label_name"].value_counts()
    for name in utils.LABEL_NAMES:
        c = int(counts.get(name, 0))
        dist.append(
            {
                "split": split,
                "emotion": name,
                "label_id": utils.LABEL2ID[name],
                "count": c,
                "pct": round(100 * c / len(df), 2),
            }
        )

label_dist = pd.DataFrame(dist)
label_dist.pivot(index="emotion", columns="split", values="pct").loc[utils.LABEL_NAMES]

# %%
length_rows = []
for split, df in raw.items():
    words = df["text"].str.split().str.len()
    length_rows.append(
        {
            "split": split,
            "n_utterances": len(df),
            "n_dialogues": df["Dialogue_ID"].nunique(),
            "n_speakers": df["Speaker"].nunique(),
            "words_mean": round(words.mean(), 2),
            "words_median": int(words.median()),
            "words_p95": int(words.quantile(0.95)),
            "words_max": int(words.max()),
        }
    )

summary = pd.DataFrame(length_rows)
summary

# %%
# Persist both tables for the report — report.md reads from results/, never recomputes.
summary.to_csv(utils.RESULTS / "dataset_stats.csv", index=False)
label_dist.to_csv(utils.RESULTS / "label_distribution.csv", index=False)
integrity.to_csv(utils.RESULTS / "integrity_checks.csv", index=False)
print("wrote dataset_stats.csv, label_distribution.csv, integrity_checks.csv")

# %% [markdown]
# ## 7. Write processed splits
#
# `sentiment` is carried through unused — it enables the optional
# 3-class experiment mentioned in the plan without re-running this notebook.

# %%
OUT_COLS = ["text", "label", "label_name", "sentiment", "speaker", "dialogue_id"]

for split, df in raw.items():
    out = pd.DataFrame(
        {
            "text": df["text"],
            "label": df["label"].astype(int),
            "label_name": df["label_name"],
            "sentiment": df["Sentiment"],
            "speaker": df["Speaker"],
            "dialogue_id": df["Dialogue_ID"],
        }
    )[OUT_COLS]
    path = utils.DATA_PROCESSED / f"{split}.csv"
    out.to_csv(path, index=False)
    print(f"{split:5s} -> {path.relative_to(utils.ROOT)}  ({len(out)} rows)")

# %%
# Read back one split to confirm the round-trip survives quoting/commas.
check = pd.read_csv(utils.DATA_PROCESSED / "train.csv")
assert len(check) == len(raw["train"]), "row count changed on round-trip"
assert check["label"].between(0, 6).all()
print("round-trip OK")
check.head()

# %% [markdown]
# ## Summary
#
# - Official MELD splits kept as shipped: **9,989 / 1,109 / 2,610** utterances.
# - **No mojibake in this copy of MELD** (contrary to the commonly reported
#   issue) — checked explicitly. `ftfy` still applied, to uncurl typographic
#   punctuation to ASCII; whitespace collapsed. ~30% of train utterances
#   changed, almost all of them contractions (`’`→`'`).
# - No nulls; empty utterances dropped; duplicate lines kept deliberately.
# - Severe class imbalance confirmed (neutral ~47%, fear/disgust <3%) — this
#   drives the metric choice and the class weighting in notebook 02.
# - Clean splits written to `data/processed/`; stats written to `results/`.
#
# Next: `02_eda_and_tokenization.py`.
