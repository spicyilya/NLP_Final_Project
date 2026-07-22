# MELD Emotion Classification — BERT vs RoBERTa

Fine-tuning and comparing `bert-base-uncased` and `roberta-base` on 7-class
emotion classification over the **MELD** dataset (text only).

**Full write-up: [`deliverables/MELD_Emotion_Classification_Report.pdf`](deliverables/MELD_Emotion_Classification_Report.pdf).**

## What this is

Multi-class text classification: given one line of dialogue from *Friends*,
predict one of `neutral, joy, surprise, anger, sadness, disgust, fear`.

The dataset is heavily imbalanced (neutral ≈ 47% of train, fear ≈ 2.7% — a
17.6× ratio), so **weighted F1 is the primary metric**, macro F1 is reported
alongside to expose rare-class performance, and training uses a class-weighted
loss.

## Repository layout

```
data/raw/          MELD CSVs (not committed — see "Data" below)
data/processed/    cleaned splits written by notebook 01
notebooks/
  01_data_preparation.py/.ipynb      load, clean, describe
  02_eda_and_tokenization.py/.ipynb  EDA, imbalance, tokenizers, class weights
  03_finetuning_and_tuning.py/.ipynb hyperparameter grid, both models
  04_evaluation_comparison.py/.ipynb test metrics, confusion, error analysis
src/
  utils.py         seed, label map, metrics, paths — shared by all notebooks
  training.py      WeightedTrainer + resumable grid runner
scripts/
  download_data.py fetch MELD CSVs
checkpoints/       best checkpoint per model (gitignored, regenerable)
results/           metrics tables, hyperparameter search CSV, figures/
deliverables/      the submitted report
```

## Deliverables

The written report is in [`deliverables/`](deliverables/):

**[`MELD_Emotion_Classification_Report.pdf`](deliverables/MELD_Emotion_Classification_Report.pdf)** — 16 pages, all 8 figures embedded.

Every number in it is read from `results/`; nothing is recomputed by hand. The
embedded figures are the same PNGs in `results/figures/`. The accompanying slide
deck is not included in this repository.

Notebooks are authored as [jupytext](https://jupytext.readthedocs.io) `.py`
files (the reviewable source of truth) and paired to `.ipynb`.

## Data

The MELD CSVs are **not committed** — MELD is released under GPL-3.0 by its
authors and this repo does not redistribute it. Fetch them with:

```bash
python scripts/download_data.py
```

This downloads `train_sent_emo.csv`, `dev_sent_emo.csv`, `test_sent_emo.csv`
(the text annotations) into `data/raw/`. We use MELD's **official splits**
unchanged — 9,989 / 1,109 / 2,610 utterances — because published baselines use
them and re-splitting would leak dialogue context across the boundary.

> Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., & Mihalcea, R.
> (2019). **MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in
> Conversations.** *Proceedings of ACL 2019.*
> Source: <https://github.com/declare-lab/MELD>

## Reproducing

### 1. Environment

Requires an NVIDIA GPU. Developed on an RTX 5070 (12 GB, Blackwell `sm_120`),
which needs a CUDA 12.x build of PyTorch — the default PyPI wheel will not work:

```bash
uv venv .venv
source .venv/Scripts/activate            # Windows (Git Bash); use .venv/bin/activate on Linux
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

Verify CUDA before doing anything else — this is the fastest failure to catch:

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.version.cuda, torch.cuda.is_bf16_supported())"
```

Register the kernel the notebooks expect:

```bash
python -m ipykernel install --user --name nlp2-venv --display-name "Python (nlp2 venv)"
```

Optionally set `HF_TOKEN` to lift the anonymous Hugging Face rate limit. Both
models are public, so it is not required:

```bash
export HF_TOKEN=...      # PowerShell: $env:HF_TOKEN = "..."
```

### 2. Run

```bash
python scripts/download_data.py
jupytext --to ipynb notebooks/*.py
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_preparation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda_and_tokenization.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_finetuning_and_tuning.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_evaluation_comparison.ipynb
```

Notebook 03 is the long one (12 GPU runs). It can also be run headless, which
is how it was actually run:

```bash
python src/training.py
```

Either way it is **resumable**: each finished run is appended to
`results/hparam_search.csv` immediately, and completed configurations are
skipped on a re-run. Killing it mid-grid costs at most one run.

Everything is seeded (`SEED = 42`; Python, NumPy, torch, Trainer, data order).
Exact GPU reproducibility is not guaranteed — cuDNN kernel selection and
non-deterministic reductions introduce small run-to-run variation.

## Method summary

| | |
|---|---|
| Models | `bert-base-uncased`, `roberta-base` (same size class → fair comparison) |
| Fine-tuning | full (all parameters) |
| Loss | class-weighted cross-entropy (inverse train frequency) |
| Grid | lr ∈ {1e-5, 2e-5, 5e-5} × batch ∈ {16, 32} = 6 runs/model, 12 total |
| Epochs | ≤5, best epoch selected on dev weighted F1 |
| Fixed | warmup_ratio 0.06, weight_decay 0.01, max_length 128, bf16, seed 42 |

The test split is used **once**, in notebook 04. Every modelling decision was
made on dev.

Results, figures, and analysis: **[`deliverables/MELD_Emotion_Classification_Report.pdf`](deliverables/MELD_Emotion_Classification_Report.pdf)**.
