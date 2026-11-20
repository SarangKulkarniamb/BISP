# Otoscopy image screening starter

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full one-week research plan, study design, WEKA/wavelet workflow, training steps, evaluation plan, team roles, and deliverable checklist.

A research and learning scaffold for otoscopic image classification. It supports a multi-class tympanic membrane classifier and a simple first-stage image quality screen. It is not a medical device and must not be used to diagnose, triage, or make treatment decisions.

## Project setup

Use Python 3.10 or newer. Install dependencies in a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Dataset layout

The downloaded Kaggle archive is stored in `data/downloads/oto-endoscopic-image-dataset.zip` and extracted to `data/raw/Oto-Endoscopic_Images/`. It contains five class folders: acute otitis media, cerumen impaction, chronic otitis media, myringosclerosis, and normal.

Create a reproducible, stratified 80/20 train/validation split with:

```powershell
python -m oto_screen.prepare_data --source data/raw/Oto-Endoscopic_Images --output data/otoscopy --val-fraction 0.2 --seed 42
```

The splitter keeps byte-identical images together. The source does not provide patient identifiers, so this is an image-level split and cannot guarantee patient-level separation. Inspect the data and account for this limitation when interpreting validation results.

For other datasets, the expected layout is class-named folders under `train/` and `val/`:

```text
data/otoscopy/
  train/
    normal/
    myringosclerosis/
    otomycosis/
    cholesteatoma/
  val/
    normal/
    myringosclerosis/
    otomycosis/
    cholesteatoma/
```

The starter uses `ImageFolder`, so class names are read from directory names. Review each dataset's license, permitted use, labels, and provenance before use. Do not commit clinical images or patient identifiers.

## Train a classifier

```powershell
python -m oto_screen.train --data data/otoscopy --output artifacts/otoscopy_resnet18.pt --epochs 10
```

This fine-tunes an ImageNet-pretrained ResNet-18. Training writes the model weights and class mapping to the output file. The initial configuration is intended for experimentation; tune it and evaluate on a genuinely independent patient-level test set before drawing conclusions.

## Predict

```powershell
python -m oto_screen.predict --model artifacts/otoscopy_resnet18.pt --image path\to\image.jpg
```

Prediction first applies a conservative image-quality heuristic based on image dimensions, brightness, and Laplacian sharpness. It can reject some unusable inputs, but it is not a learned quality model and has not been validated for clinical capture conditions. Adjusting its thresholds requires a labeled quality dataset and evaluation.

## Evaluate a saved model

```powershell
python -m oto_screen.evaluate --model artifacts/otoscopy_resnet18.pt --data data/otoscopy/val --output-dir results/validation
```

This writes per-class and macro metrics to `results/validation/metrics.json` and the confusion matrix to `results/validation/confusion_matrix.csv`. These are validation metrics: the training script uses this split to select its best checkpoint, so they are not an independent final test estimate.

## Limitations and next steps

- This is an educational prototype, not a validated clinical system.
- Public datasets may be small, imbalanced, duplicated, or collected with different devices and populations. Check these risks before training.
- Report per-class sensitivity, specificity, precision, calibration, and confusion matrices on held-out patients; include uncertainty and subgroup/device analysis where data permits.
- Add a separately trained quality classifier only after collecting representative labels for acceptable, blurry, obstructed, and poorly lit images.
- Obtain appropriate ethics, privacy, and regulatory review before any prospective or patient-facing use.
