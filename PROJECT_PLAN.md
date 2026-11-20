# Mini Research Project Plan

## Project title

**Wavelet-Enhanced Machine Learning for Five-Class Otoscopic Image Classification**

## Project at a glance

- **Signal type:** 2D biomedical images (oto-endoscopic / otoscopic photographs).
- **Task:** classify images into the five classes actually present in the downloaded dataset.
- **Dataset:** Kaggle, *Oto-Endoscopic Image Dataset*, 3,014 JPEG images, about 91 MB, marked CC0 on its data card.
- **Classes in the extracted folders:** Acute Otitis Media (601), Cerumen Impaction (602), Chronic Otitis Media (602), Myringosclerosis (605), and Normal (604).
- **Proposed classical method:** multilevel 2D Discrete Wavelet Transform (DWT) texture descriptors, optionally fused with GLCM texture descriptors, classified in WEKA.
- **Deep-learning comparator:** the project's fine-tuned ResNet-18 in PyTorch.
- **Important scope correction:** this downloaded dataset does not contain otomycosis or cholesteatoma labels. Do not claim the model detects those conditions. “Cavity after cholesteatoma removal” in a different paper is not the same as a cholesteatoma class.
- **Current status:** archive and extracted files are present; an image-level 80/20 train/validation split exists. The model has not been trained, and the code does not yet produce the DWT/WEKA experiments or a separate final test set.

## Research question and hypothesis

**Research question:** On a fixed, held-out split of this five-class dataset, does a DWT-based feature pipeline classified with WEKA improve macro-F1 and the weakest-class sensitivity over non-wavelet handcrafted-feature baselines and the existing ResNet-18 baseline?

**Hypothesis to test (not a promised result):** multiscale wavelet texture descriptors, optionally fused with GLCM descriptors, may improve recognition of texture-dependent classes such as myringosclerosis. The result must be judged on the predeclared test split. If it does not improve, report that accurately and discuss why.

## Inputs and expected outputs

### Inputs

1. Raw dataset: `data/raw/Oto-Endoscopic_Images/<class_name>/*.jpg`.
2. For the existing ResNet experiment: `data/otoscopy/train/<class_name>/` and `data/otoscopy/val/<class_name>/`.
3. For WEKA: one feature record per image, with numerical features and a nominal class label. Image path/name must be kept in a separate sidecar for error review, not used as a predictive feature.
4. For single-image inference: an image file and a trained checkpoint.

### Outputs to save

- Dataset manifest and split counts: `results/data_manifest.csv` and `results/split_summary.csv`.
- DWT/GLCM features in matching ARFF files: `results/features/train.arff`, `validation.arff`, and `test.arff`.
- Saved WEKA models and experiment settings: `artifacts/weka/`.
- ResNet checkpoint: `artifacts/otoscopy_resnet18.pt`.
- Metrics, confusion matrices, ROC curves, and error examples: `results/`.
- Final report, slide deck, source code, and references: `deliverables/`.

## Experimental design

### 1. Dataset audit

- Preserve the original archive and raw folder. Never edit the raw files in place.
- Record file count, class count, image dimensions, corrupt/unreadable files, and duplicate or near-duplicate content.
- Use the actual folder counts as the class counts; the Kaggle page's prose contains an inconsistent statement about balance.
- Keep labels exactly aligned with the folders. Treat the task as five-class classification, not normal-vs-disease unless a separate binary endpoint is explicitly declared.
- The dataset does not supply patient identifiers. Any split is therefore image-level; duplicate grouping helps, but cannot guarantee patient-level independence. Search for near duplicates before final evaluation and disclose the limitation.

### 2. Use a three-way split for the final study

The currently prepared `train/val` split is useful for a pilot. It is not sufficient for an unbiased final score because `train.py` checks validation accuracy every epoch and saves the best checkpoint using that same validation set.

For final comparisons, make a new stratified **70% train / 15% validation / 15% test** split, with fixed seed `42`; keep exact duplicate groups together. Use train for fitting, validation for model/threshold selection, and test once for final metrics. Do not change parameters after looking at test results. If a patient-level or device-level identifier becomes available, use it to group the split instead of image-level sampling. The present dataset does not provide such identifiers.

### 3. Preprocessing

- Keep a copy of each original image and record the preprocessing choices.
- For the WEKA feature pipeline, convert to grayscale and resize consistently (start with 128 × 128 pixels); do not use any test-set data to choose preprocessing parameters.
- For the ResNet-18 comparator, retain RGB, resize to 224 × 224, normalize with ImageNet mean and standard deviation, and apply mild training-only augmentation. Do not augment validation or test images.
- Fit any data-dependent scaling/feature selection using training data only, then apply the fitted transform to validation/test.

### 4. Wavelet and WEKA method

1. Compute a 2D DWT using Haar or `db2` at levels 1–3 on each preprocessed grayscale image.
2. For each detail sub-band (LH, HL, HH) at each level, extract coefficient energy, mean absolute coefficient, standard deviation, and entropy. Include approximation-band summaries if useful.
3. Optional feature-fusion arm: add GLCM contrast, homogeneity, energy, and correlation calculated using fixed gray-level quantization and fixed pixel distances/angles. Document those settings.
4. Export numeric features with a nominal five-class label to ARFF. Use the same declared attributes and class value order for train, validation, and test. Exclude filename, directory, image dimensions if they encode collection cues, and all identifiers from model inputs.
5. In WEKA Explorer, load the training ARFF, set `class` as the class attribute, and compare a small predeclared set: RandomForest (primary), SMO/SVM, J48, and Logistic. Tune only with stratified cross-validation on training data. Save the selected model and settings. Then evaluate once on the test ARFF.
6. Compare feature ablations: DWT only; GLCM only; DWT + GLCM. The proposed method is DWT + GLCM with RandomForest only if the team predeclares that as the primary method; otherwise choose DWT + RandomForest as primary and treat fusion as an ablation.

Use ARFF rather than separate inferred CSV schemas where possible: WEKA's own documentation notes that CSV may infer nominal class values differently between train and test files.

### 5. ResNet-18 comparator

The current scaffold fine-tunes an ImageNet-pretrained ResNet-18 on RGB images. Train with training images, select the best checkpoint using validation only, then compute final metrics on the untouched test split. Save the checkpoint, class ordering, seed, epoch count, learning rate, and hardware details. Compare model families on exactly the same final test images.

### 6. Baselines and comparisons

Run these local experiments using the same split:

| ID | Method | Purpose |
|---|---|---|
| B1 | WEKA RandomForest on GLCM only | Non-wavelet handcrafted baseline |
| B2 | WEKA RandomForest on DWT only | Measures wavelet contribution |
| P1 | WEKA RandomForest on DWT + GLCM | Proposed feature-fusion method |
| DL1 | Fine-tuned ResNet-18 on RGB | End-to-end comparator |

Optionally add WEKA SMO/J48/Logistic on the same primary features, but keep the comparison small enough to finish and explain.

### 7. Metrics and statistical reporting

Report the confusion matrix and **per-class** sensitivity/recall, specificity, precision, and F1. Also report macro averages so each class has equal weight, plus accuracy and macro one-vs-rest ROC-AUC from class scores/probabilities. State the averaging convention. Include 95% confidence intervals where feasible (e.g., stratified bootstrap over test images), and avoid treating image-level intervals as patient-level uncertainty.

For each primary comparison, report the difference in macro-F1 and macro sensitivity on the same test images. Improvement is supported only if the comparison is methodologically fair and the measured result supports it. A high score alone does not establish clinical usefulness or generalization.

### 8. Literature comparison

Use recent papers to motivate the method and explain limitations. Published figures below come from different datasets, labels, splits, and devices; they are context, **not a leaderboard** and not a direct pass/fail threshold for this experiment.

| Year and paper | Method / task | Reported result or relevance | How this project should use it |
|---|---|---|---|
| 2024, Dubois et al., *npj Digital Medicine* | Smartphone-based Inception-v2, 11 diagnostic classes; a held-out test set was used | Binary normal/abnormal test sensitivity 99.0%, specificity 95.2%; performance varied by class | Cite as a large-scale smartphone otoscopy reference. Its 11 labels and data volume are different from this five-class Kaggle set. |
| 2025, Guo et al., *Medicine* | Compared ResNet-18, GoogLeNet, AlexNet, MobileNet-V3, VGG-19 on 819 images / 4 OM classes | VGG-19 reported accuracy 94.51%, sensitivity 94.18%, specificity 98.10%, precision 93.86% | Use as a recent CNN benchmark example; do not claim direct superiority on a different dataset. |
| 2025, Seo et al., *Journal of Clinical Medicine* | Multi-stage quality, segmentation, laterality, and disease pipeline on 2,964 images / 4 OM classes | Reported ConvNeXt disease accuracy 88.7%, with class F1 values including AOM 0.78 and COM 0.92 | Discuss why an end-to-end workflow includes image quality and membrane localization. Dataset/task differs. |
| 2026, Xu et al., *PLOS ONE* | Examined dataset bias and external generalization across three public otoscopy datasets | Found that high internal AUC could coexist with poor external generalization and reliance on non-clinical artifacts | Use to justify duplicate/artifact review, careful splits, and cautious claims. |
| 2026, BioOtoFusionNet, *Engineering Science and Technology, an International Journal* | DWT frequency stream plus shape stream for four-class otoscopy | The paper reports 96.8% accuracy, 95.6% F1, and 98.4% AUC-ROC | The closest wavelet-based comparison; use as methodological context, but its dataset/task and reported protocol are not identical. |

### 9. Error and bias analysis

- Review the test confusion matrix and a small, representative set of correct and incorrect predictions with class labels.
- Check whether the model appears to use image borders, lighting, device framing, labels/watermarks, or backgrounds instead of ear anatomy. Use Grad-CAM for ResNet as a qualitative aid only; it is not proof of clinical reasoning.
- Record limitations: one public source, no patient/device metadata, small dataset, uncertain label provenance, class/task mismatch with intended claims, and internal image-level evaluation.
- Do not call the quality heuristic a learned quality model. The dataset has no quality labels, so the heuristic is not part of the main disease-classification experiment.

## Run the current ResNet pilot

Open PowerShell in the project folder and run:

```powershell
cd C:\Users\admin\Documents\ChatGPT\BISP
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

The dataset split is already prepared; do not run the split command again with the same output path. Start the pilot model with:

```powershell
python -m oto_screen.train --data data/otoscopy --output artifacts/otoscopy_resnet18.pt --epochs 10
```

The command prints training loss and validation accuracy per epoch and saves the best validation checkpoint to `artifacts/otoscopy_resnet18.pt`. The checkpoint is a model artifact, not a report of validated clinical performance.

After training, try one image:

```powershell
python -m oto_screen.predict --model artifacts/otoscopy_resnet18.pt --image path\to\image.jpg
```

The output is JSON. If the basic quality heuristic rejects the image, it reports `rejected_for_quality` and reasons. Otherwise it returns ranked class scores under `screening_result`. Scores are model outputs, not calibrated probabilities unless calibration is separately implemented and evaluated. This is a research prototype and must not guide patient care.

## One-week schedule

| Day | Work | End-of-day evidence |
|---|---|---|
| 1 | Confirm exact scope/classes; audit archive, counts, duplicates, and licenses; download/read the selected papers | Signed-off question, dataset manifest, bibliography |
| 2 | Create a fixed train/validation/test split; document preprocessing; implement DWT and GLCM feature extraction | Split summary, feature files, reproducible settings |
| 3 | Run WEKA feature baselines and ablations; save models and settings | WEKA results, confusion matrices |
| 4 | Train the ResNet-18 baseline; select only on validation; run final test once | Checkpoint, metrics, loss curves |
| 5 | Compare models; inspect errors, duplicates, and visual artifacts; calculate intervals | Results tables, error analysis, limitations |
| 6 | Draft report and slides from actual outputs; cite every borrowed idea/result; check originality | Complete draft PDF/PPT and source bundle |
| 7 | Re-run documented commands; verify files/references; run institution-approved similarity and AI-use checks; rehearse | Final PDF, PPT, code, results, team contribution statement |

## Three-person team split

1. **Data and wavelets:** dataset audit, split manifest, preprocessing, DWT/GLCM extraction, WEKA ARFF generation.
2. **Models and evaluation:** WEKA runs, ResNet baseline, metric calculations, plots, saved configurations.
3. **Research and deliverables:** literature table, report/PPT integration, citation checks, reproducibility checklist, meeting coordination.

All members should understand the final method and review the results; each member should contribute to interpretation rather than only formatting.

## Report and presentation checklist

### PDF report

1. Abstract and keywords.
2. Problem statement, research question, and contribution.
3. Dataset source, license, class counts, audit, split, and ethical/safety boundary.
4. DWT/GLCM feature definitions and WEKA settings.
5. ResNet baseline and training details.
6. Evaluation metrics and fair comparison protocol.
7. Results: per-class and macro table, confusion matrices, ROC curves, and uncertainty.
8. Discussion: whether the hypothesis was supported, comparison with 2024–2026 literature, errors, bias, and limitations.
9. Conclusion and next steps.
10. References and team contributions.

### PowerPoint

Aim for 8–10 slides: motivation; research question; dataset; pipeline; wavelet illustration; WEKA/model setup; results; literature comparison; limitations; conclusion/demo.

### Academic integrity

Write the report in the team's own words and cite papers, software, and datasets. Keep experiment logs and original plots. Do not invent results or use published-paper scores as if obtained on this dataset. Follow the institution's rules for AI assistance and similarity checking; review any flagged passages manually and correct missing citations or overly close paraphrases before submission. Detection tools are imperfect and cannot certify authorship by themselves.

## References (2024–2026)

1. Dubois C, Eigen D, Simon F, et al. Development and validation of a smartphone-based deep-learning-enabled system to detect middle-ear conditions in otoscopic images. *npj Digital Medicine*. 2024;7:162. https://doi.org/10.1038/s41746-024-01159-9
2. Guo Q, Xie L, Zhou L. Deep learning for otitis media classification using otoscopic image. *Medicine (Baltimore)*. 2025;104(48):e46218. https://doi.org/10.1097/MD.0000000000046218
3. Seo HW, Ko DW, Oh J, et al. Development and Validation of a CNN-Based Diagnostic Pipeline for the Diagnosis of Otitis Media. *Journal of Clinical Medicine*. 2025;14(23):8572. https://doi.org/10.3390/jcm14238572
4. Xu et al. Towards reliable use of artificial intelligence to classify otitis media using otoscopic images: Addressing bias and improving data quality. *PLOS ONE*. 2026;21(5):e0338867. https://doi.org/10.1371/journal.pone.0338867
5. BioOtoFusionNet – A bio-attention & frequency-aware hybrid network for ear disease classification. *Engineering Science and Technology, an International Journal*. 2026;77:102354. https://doi.org/10.1016/j.jestch.2026.102354

### Dataset and software references

- Kaggle dataset: https://www.kaggle.com/datasets/ucimachinelearning/oto-endoscopic-image-dataset
- WEKA ARFF/CSV format guidance: https://waikato.github.io/weka-wiki/formats_and_processing/arff/ and https://waikato.github.io/weka-wiki/faqs/use_csv_files/
