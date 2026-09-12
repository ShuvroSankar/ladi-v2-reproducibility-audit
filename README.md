# LADI v2 Reproducibility & Distribution-Shift Audit

An independent audit of MIT Lincoln Laboratory's [LADI v2](https://arxiv.org/abs/2406.02780) disaster-imagery classification benchmark. What started as an attempt to reproduce and improve on the released BiT-50 baseline turned into a full reproducibility investigation: the released checkpoint could not be made to match the paper's reported numbers despite an exhaustive audit, and the real val/test performance gap turns out to be driven by a much harder problem than simple class imbalance — a complete, zero-overlap split by disaster event.

## Key Findings

- **Reproduction gap:** the official BiT-50 reference checkpoint reproduces at 81.1% validation / 61.1% test mAP, well below the paper's reported 93.3% / 91.1% — a gap that survives a 9-point audit (label integrity, image preprocessing, checkpoint loading, an exact 4-decimal match to the model card's own published example) and remains unresolved.
- **Zero event overlap:** train, validation, and test share no disaster events in common, in either direction. Every test image comes from a 2023 event the model has never seen in any form.
- **Leave-one-event-out cross-validation** (5/5 folds): mean 0.891 mAP, SD 0.044 — well above the real test mAP, showing a held-out *2022* event is still far easier than the actual 2023 test set.
- **Domain-classifier analysis:** a simple linear probe on the model's own backbone features separates 2022 from 2023 images with AUC 0.893 — direct evidence the shift is visual and measurable, not just "unseen data."
- **Three interventions tested:** loss reweighting (no significant effect), targeted augmentation (a measurable *negative* effect, mean Δ −0.031), and per-class threshold tuning (a naive version catastrophically miscalibrates one class; a guarded version yields a small, real, zero-regression gain of +0.0061 mean F1).

Full methodology, tables, and figures are in the accompanying paper (`ladi_v2_audit_paper.pdf`).

## Repository Contents

| File | Description |
|---|---|
| `pipeline.ipynb` | Baseline reproduction pipeline — streams the official dataset from Hugging Face, loads the released BiT-50 checkpoint, computes val/test mAP. |
| `kaggle_finetuning_experiments.ipynb` | All fine-tuning and diagnostic work: focal loss / weighted BCE experiments, event-overlap analysis, leave-one-event-out CV, domain-classifier AUC, augmentation, and threshold tuning. |
| `make_charts.py` | Regenerates the paper's comparison figures from the recorded results (matplotlib). |
| `ladi_v2_audit_paper.pdf` | Full write-up: methodology, results, discussion, and references. |

*(Rename to match whatever you actually upload — these are suggested names matching what each notebook does.)*

## Background

LADI v2 is a multi-label disaster-imagery dataset built from Civil Air Patrol aerial photography, with a deliberate train (2015–2022) / test (2023) split intended as a realistic distribution-shift benchmark. See the [original paper](https://arxiv.org/abs/2406.02780) and [dataset repository](https://github.com/LADI-Dataset/ladi-overview) for details on data collection and labeling.

## Reproducing This Work

1. Dataset: [MITLL/LADI-v2-dataset](https://huggingface.co/datasets/MITLL/LADI-v2-dataset) on Hugging Face, or the resized CSV/image release linked from the dataset repository.
2. Model: [MITLL/LADI-v2-classifier-small-reference](https://huggingface.co/MITLL/LADI-v2-classifier-small-reference) (BiT-50), loaded via `transformers`.
3. `pipeline.ipynb` runs anywhere with the HF `datasets` and `transformers` libraries. `kaggle_finetuning_experiments.ipynb` was run on Kaggle's T4 GPU tier and expects the dataset mounted as a Kaggle input — adjust `CSV_PATH` / `IMAGE_ROOT` for a different environment.

## Citation

If you use this audit, please also cite the original dataset:
<!-- 
```bibtex
@article{scheele2024ladi,
  title={LADI v2: Multi-label Dataset and Classifiers for Low-Altitude Disaster Imagery},
  author={Scheele, Samuel and Picchione, Katherine and Liu, Jeffrey},
  journal={arXiv preprint arXiv:2406.02780},
  year={2024}
}
```
-->

## Author

Shuvro Sankar Sen — American International University-Bangladesh
Supervisor: Prof. Dr. M. Shamim Kaiser

## Acknowledgments

Built on data and models released by MIT Lincoln Laboratory under the terms described in the original LADI v2 repository. This is an independent, third-party audit and is not affiliated with or endorsed by the original authors.

