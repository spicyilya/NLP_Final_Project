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

| Model | weighted F1 | macro F1 | accuracy |
|---|---|---|---|
| majority baseline | 0.313 | 0.093 | 0.481 |
| bert-base-uncased | 0.593 | 0.434 | 0.576 |
| **roberta-base** | **0.605** | **0.460** | **0.591** |

- Both models ≈ **5× the baseline's macro F1**, while the baseline already gets
  48% *accuracy* — that contrast is the whole argument for the metric choice.
- Inside the published text-only MELD range (~57–65% weighted F1).
- **The punchline:** BERT won dev (0.603 vs 0.584), RoBERTa won test
  (0.605 vs 0.593). **The ranking flipped.** One seed, ~0.01–0.02 gaps ⇒ the
  honest claim is "not separated", not "RoBERTa wins".

> This is the slide to be brave on. Reporting a flipped ranking is a better
> result than pretending you found a winner.

## Slide 5 — Analysis: where it breaks (~60s)

- Figure: `results/figures/per_class_f1.png` + `confusion_matrices.png`.
- Performance tracks frequency: neutral 0.73 → fear 0.14–0.22.
- **Every** top confusion for both models is `X → neutral`. Rare emotions
  collapse into the majority class.
- Class weighting worked *and* overcorrected: RoBERTa fear recall 0.400 but
  precision **0.154** — 5 of 6 "fear" predictions are wrong. ~38% of all errors
  are neutral-misread-as-emotion vs ~18% the other way.
- **The refuted hypothesis** (good slide material): we expected short utterances
  to be hardest. They're **easier** — 0.660 vs 0.544 accuracy. Not a class-mix
  artifact (macro F1 also higher; neutral share equal).
- **The real ceiling:** 24 strings appear in test with *different gold labels*
  (5.9% of test). `"Hey!"` is labelled anger, joy, neutral, sadness *and*
  surprise. Identical input can't yield two outputs — provably unfixable
  from text.

## Slide 6 — Conclusions (~40s)

- Pick **RoBERTa** — but for macro F1 / rare classes, and the gap is not settled.
- Metric choice mattered more than any hyperparameter (lr spread 0.018, batch
  0.015; best-epoch selection alone was worth 0.017).
- With more time, in order: **multiple seeds + significance test** (nothing else
  matters until the gap is real), **dialogue context** (attacks the ceiling
  above), softer class weighting, then the multimodal signals MELD ships.

---

## Recording notes

- Everything on screen comes from `results/` — no slide recomputes anything.
- 5 minutes is tight: slide 2 (imbalance) and slide 5 (limitations) are the
  ones that show understanding. Slides 1 and 3 can be brisk.
- Don't oversell the winner if the gap is small — say so, and say what would
  settle it (multiple seeds).
