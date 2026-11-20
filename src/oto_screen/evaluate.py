"""Evaluate a saved classifier and write metrics plus a confusion matrix."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via average ranks, with tie handling and no sklearn dependency."""
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = ranks[y_true].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="ImageFolder class directory, e.g. data/otoscopy/val")
    parser.add_argument("--output-dir", type=Path, default=Path("results/validation"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    transform = transforms.Compose([
        transforms.Resize((checkpoint.get("input_size", 224),) * 2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(args.data, transform=transform)
    if dataset.classes != classes:
        raise ValueError(f"Dataset classes {dataset.classes} do not match checkpoint classes {classes}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    all_true, all_probabilities = [], []
    with torch.no_grad():
        for images, labels in loader:
            probabilities = model(images.to(device)).softmax(1).cpu().numpy()
            all_probabilities.append(probabilities)
            all_true.append(labels.numpy())
    y_true = np.concatenate(all_true)
    probabilities = np.concatenate(all_probabilities)
    y_pred = probabilities.argmax(1)

    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    per_class = {}
    auc_values = []
    for index, name in enumerate(classes):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
        auc = binary_auc(y_true == index, probabilities[:, index])
        if np.isfinite(auc):
            auc_values.append(auc)
        per_class[name] = {
            "support": int((y_true == index).sum()),
            "sensitivity_recall": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1_score": f1,
            "auc_ovr": auc,
        }

    split_name = args.data.name.lower()
    note = ("Held-out test split; it was not used for model fitting or checkpoint selection. "
            "The split is image-level because patient identifiers are unavailable.") if split_name == "test" else (
            "Validation metrics only; this split may have been used to select the best checkpoint and is not an independent test estimate.")
    metrics = {
        "evaluation_split": args.data.as_posix(),
        "number_of_images": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_sensitivity_recall": float(np.mean([v["sensitivity_recall"] for v in per_class.values()])),
        "macro_specificity": float(np.mean([v["specificity"] for v in per_class.values()])),
        "macro_precision": float(np.mean([v["precision"] for v in per_class.values()])),
        "macro_f1_score": float(np.mean([v["f1_score"] for v in per_class.values()])),
        "macro_auc_ovr": float(np.mean(auc_values)) if auc_values else None,
        "classes": classes,
        "per_class": per_class,
        "confusion_matrix_rows_actual_columns_predicted": matrix.tolist(),
        "note": note,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    with (args.output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual\\predicted", *classes])
        for class_name, row in zip(classes, matrix.tolist()):
            writer.writerow([class_name, *row])

    split_label = "Test" if split_name == "test" else "Validation"
    print(f"{split_label} metrics for {len(y_true)} images (device={device}):")
    for key in ("accuracy", "macro_sensitivity_recall", "macro_specificity", "macro_precision", "macro_f1_score", "macro_auc_ovr"):
        print(f"{key}: {metrics[key]:.4f}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print("actual\\predicted\t" + "\t".join(classes))
    for class_name, row in zip(classes, matrix.tolist()):
        print(class_name + "\t" + "\t".join(map(str, row)))
    print(f"Saved metrics to {args.output_dir / 'metrics.json'}")
    print(f"Saved confusion matrix to {args.output_dir / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
