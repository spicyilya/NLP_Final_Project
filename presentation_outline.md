# 5-minute presentation — slide skeleton

Record last, from the figures already in `results/figures/`. Six slides,
~50 seconds each. Numbers are filled in from `results/` — see `report.md`.

---

## Slide 1 — The task (~40s)

- Multi-class text classification: one line of *Friends* dialogue → one of
  **7 emotions** (neutral, joy, surprise, anger, sadness, disgust, fear).
- Dataset: **MELD**, text annotations only. Official splits: 9,989 / 1,109 / 2,610.
- Compare **`bert-base-uncased` vs `roberta-base`** under an identical protocol.

> Talking point: text-only is the assignment's scope — and it's also the source
> of the ceiling we hit at the end. Flag it early so slide 5 pays off.

## Slide 2 — The problem is imbalance (~50s)

- Figure: `results/figures/class_distribution.png`
- **neutral = 47% of train; fear = 2.7%. A 17.6× ratio.**
- "Always predict neutral" scores **47% accuracy** and is useless.
- ⇒ Primary metric is **weighted F1**, with **macro F1** to expose rare classes.
- ⇒ Loss is **class-weighted** (inverse train frequency), not resampled —
  oversampling ~270 fear examples 17× invites memorization.

> This is the slide that justifies every later decision. Don't rush it.

## Slide 3 — Method (~50s)

- Full fine-tuning, 7-way head, bf16, seed 42, max_length 128 (truncates zero
  utterances; dynamic padding keeps it cheap).
- Grid: **lr {1e-5, 2e-5, 5e-5} × batch {16, 32} = 6 runs/model, 12 total**,
  ≤5 epochs with best-epoch selection on dev.
- Fixed: warmup 0.06, weight decay 0.01 — so the grid isolates lr × batch size.
- Identical protocol for both models ⇒ differences come from **pretraining**,
  not tuning effort or capacity.
- Optional 15s: tokenizer contrast — `PIVOT!` → BERT `pi ##vot` (casing thrown
  away) vs RoBERTa `P IV OT` (casing kept).

## Slide 4 — Results (~60s)

- Table: `results/test_metrics.csv` — both models + majority baseline.
- Figure: `results/figures/confusion_matrices.png`.
- State plainly: which model wins on weighted F1, on macro F1, and by how much.
- Anchor against the published text-only MELD range (~57–65% weighted F1).

## Slide 5 — Analysis: where it breaks (~60s)

- Figure: `results/figures/per_class_f1.png`.
- Top confusion pairs (from `results/confusion_pairs.csv`).
- The honest limitation: median utterance is **6 words**. Lines like "What?"
  and "Hey." have no emotion *in the text* — the label lives in delivery and
  dialogue context the model never sees.
- Hard evidence, not hand-waving: identical strings appear in test with
  **different gold labels**. No text-only model can get both right.
- Accuracy on ≤3-word utterances vs longer ones.

## Slide 6 — Conclusions (~40s)

- Which model to pick, and the honest size of the gap.
- Class weighting did / didn't buy rare-class recall (per-class evidence).
- With more time: dialogue context (previous utterances), speaker embeddings,
  and the multimodal signals MELD ships but this study ignores.

---

## Recording notes

- Everything on screen comes from `results/` — no slide recomputes anything.
- 5 minutes is tight: slide 2 (imbalance) and slide 5 (limitations) are the
  ones that show understanding. Slides 1 and 3 can be brisk.
- Don't oversell the winner if the gap is small — say so, and say what would
  settle it (multiple seeds).
