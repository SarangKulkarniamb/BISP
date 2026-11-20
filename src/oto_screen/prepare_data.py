"""Create a reproducible stratified train/validation split from class folders."""

import argparse
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_class(files: list[Path], validation_fraction: float, test_fraction: float,
                rng: random.Random):
    groups = defaultdict(list)
    for path in files:
        groups[file_hash(path)].append(path)
    grouped_files = list(groups.values())
    rng.shuffle(grouped_files)
    target_validation_count = max(1, round(len(files) * validation_fraction))
    target_test_count = round(len(files) * test_fraction)
    validation, test, training = [], [], []
    for group in grouped_files:
        if len(test) < target_test_count and len(training) + len(validation) + len(group) < len(files):
            test.extend(group)
        elif len(validation) < target_validation_count and len(training) + len(validation) + len(test) + len(group) < len(files):
            validation.extend(group)
        else:
            training.extend(group)
    if not validation or not training or (test_fraction > 0 and not test):
        raise ValueError("Each class needs enough distinct images to place images in every requested split")
    return training, validation, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Folder containing one subfolder per class")
    parser.add_argument("--output", type=Path, default=Path("data/otoscopy"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.0,
                        help="Optional held-out test fraction; 0 keeps the original train/val layout")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.val_fraction < 1 or not 0 <= args.test_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1 and --test-fraction between 0 and 1")
    if args.val_fraction + args.test_fraction >= 1:
        parser.error("--val-fraction plus --test-fraction must be less than 1")
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        parser.error(f"Source folder does not exist: {source}")
    if output.exists():
        parser.error(f"Output already exists; move or remove it first: {output}")

    classes = sorted(path for path in source.iterdir() if path.is_dir())
    if not classes:
        parser.error(f"No class folders found in {source}")

    rng = random.Random(args.seed)
    plan = {}
    cross_class_hashes = {}
    for class_dir in classes:
        images = sorted(path for path in class_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if len(images) < 2:
            parser.error(f"Class {class_dir.name} needs at least two images")
        for image in images:
            digest = file_hash(image)
            previous = cross_class_hashes.get(digest)
            if previous is not None and previous != class_dir.name:
                parser.error(f"Identical image content has conflicting labels: {previous} and {class_dir.name}")
            cross_class_hashes[digest] = class_dir.name
        plan[class_dir.name] = split_class(images, args.val_fraction, args.test_fraction, rng)

    for class_name, (training, validation, test) in plan.items():
        splits = [("train", training), ("val", validation)]
        if args.test_fraction > 0:
            splits.append(("test", test))
        for split_name, files in splits:
            destination = output / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)
            for source_file in files:
                shutil.copy2(source_file, destination / source_file.name)

    print(f"Created split in {output} (seed={args.seed}, validation_fraction={args.val_fraction}, test_fraction={args.test_fraction})")
    for class_name, (training, validation, test) in plan.items():
        counts = f"{class_name}: train={len(training)}, val={len(validation)}"
        if args.test_fraction > 0:
            counts += f", test={len(test)}"
        print(counts)
    print("Exact duplicate images were kept together. The split is image-level; patient IDs are not available.")


if __name__ == "__main__":
    main()
