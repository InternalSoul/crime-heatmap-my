from __future__ import annotations

import argparse
import csv
import random
import shutil
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_DATASETS_ROOT = Path(__file__).resolve().parent.parent / "datasets"


def normalize_label(value: str) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())


def discover_dataset_classes(datasets_root: Path) -> set[str]:
    classes: set[str] = set()
    if not datasets_root.exists():
        return classes

    for csv_path in datasets_root.rglob("*.csv"):
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                fields = {name.lower(): name for name in reader.fieldnames}
                class_key = next((fields[key] for key in ("class", "label", "name", "category") if key in fields), None)
                if class_key is None:
                    continue
                for row in reader:
                    label = normalize_label(row.get(class_key, ""))
                    if label:
                        classes.add(label)
        except (OSError, UnicodeDecodeError, csv.Error):
            continue

    for xml_path in datasets_root.rglob("*.xml"):
        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError):
            continue

        for elem in root.findall(".//*[@label]"):
            label = normalize_label(elem.attrib.get("label", ""))
            if label:
                classes.add(label)

        for label_node in root.findall(".//label"):
            name_node = label_node.find("name")
            if name_node is not None and name_node.text:
                label = normalize_label(name_node.text)
                if label:
                    classes.add(label)

    return classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO dataset from many images and videos (frame extraction)."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder containing data/raw and where output data/yolo_dataset will be created.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=12,
        help="Extract one frame every N frames from each video.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=350,
        help="Hard limit for extracted frames per video.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Maximum number of videos to process (0 means all).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )
    parser.add_argument(
        "--auto-label",
        action="store_true",
        help="Use a pretrained YOLO model to auto-generate labels.",
    )
    parser.add_argument(
        "--auto-label-model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained YOLO model used for auto labeling.",
    )
    parser.add_argument(
        "--auto-label-conf",
        type=float,
        default=0.35,
        help="Confidence threshold for auto labels.",
    )
    parser.add_argument(
        "--auto-label-batch-size",
        type=int,
        default=16,
        help="Number of images to send through YOLO at once during auto labeling.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="person,knife,baseball bat,scissors,handbag,backpack",
        help="Comma separated object class names to keep during auto labeling.",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=DEFAULT_DATASETS_ROOT,
        help="Optional external datasets folder to include media/classes from.",
    )
    return parser.parse_args()


def extract_frames(
    video_path: Path,
    destination_dir: Path,
    frame_stride: int,
    max_frames: int,
) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return []

    extracted_paths: list[Path] = []
    frame_index = 0
    saved = 0
    stem = video_path.stem.replace(" ", "_")

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % frame_stride == 0:
            output_path = destination_dir / f"{stem}_f{frame_index:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            extracted_paths.append(output_path)
            saved += 1
            if saved >= max_frames:
                break

        frame_index += 1

    cap.release()
    print(f"[INFO] Extracted {len(extracted_paths)} frames from {video_path.name}")
    return extracted_paths


def find_media_files(root: Path, extensions: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)
    return sorted(files)


def copy_existing_label_if_present(source_image: Path, target_label_path: Path) -> None:
    source_label = source_image.with_suffix(".txt")
    if source_label.exists():
        shutil.copy2(source_label, target_label_path)
    else:
        target_label_path.write_text("", encoding="utf-8")


def is_readable_image(image_path: Path) -> bool:
    image = cv2.imread(str(image_path))
    return image is not None and image.size > 0


def auto_label_images(
    image_paths: list[Path],
    label_paths: list[Path],
    model_name: str,
    classes_to_keep: set[str],
    conf: float,
    batch_size: int,
) -> tuple[dict[int, str], int]:
    from ultralytics import YOLO

    print(f"[INFO] Loading auto-label model: {model_name}")
    model = YOLO(model_name)

    discovered: dict[int, str] = {}
    labeled_count = 0
    started_at = time.monotonic()
    total_images = len(image_paths)

    if total_images == 0:
        return discovered, labeled_count

    print(f"[INFO] Auto-labeling {total_images} images in batches of {batch_size}...")

    processed = 0
    for start_index in range(0, total_images, batch_size):
        batch_images = image_paths[start_index : start_index + batch_size]
        batch_labels = label_paths[start_index : start_index + batch_size]
        readable_pairs: list[tuple[Path, Path]] = []

        for image_path, label_path in zip(batch_images, batch_labels):
            if is_readable_image(image_path):
                readable_pairs.append((image_path, label_path))
            else:
                label_path.write_text("", encoding="utf-8")

        if readable_pairs:
            batch_sources = [str(image_path) for image_path, _ in readable_pairs]
            results = model.predict(source=batch_sources, conf=conf, verbose=False)

            for (image_path, label_path), result in zip(readable_pairs, results):
                lines: list[str] = []
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xywhn.cpu().tolist()
                    classes = result.boxes.cls.cpu().tolist()

                    for xywh, cls_idx_float in zip(boxes, classes):
                        cls_idx = int(cls_idx_float)
                        class_name = str(model.names.get(cls_idx, cls_idx)).lower()
                        if class_name not in classes_to_keep:
                            continue

                        discovered[cls_idx] = str(model.names.get(cls_idx, cls_idx))
                        x, y, w, h = xywh
                        lines.append(f"{cls_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

                label_path.write_text("\n".join(lines), encoding="utf-8")
                if lines:
                    labeled_count += 1

        processed += len(batch_images)
        elapsed = time.monotonic() - started_at
        print(f"[INFO] Auto-label progress: {processed}/{total_images} images in {elapsed:.1f}s")

    return discovered, labeled_count


def write_dataset_yaml(dataset_root: Path, names_map: dict[int, str]) -> Path:
    sorted_items = sorted(names_map.items(), key=lambda item: item[0])
    if not sorted_items:
        raise RuntimeError(
            "No classes discovered for dataset.yaml. Add labels manually or use --auto-label."
        )

    index_remap = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(sorted_items)}

    for split in ("train", "val"):
        labels_dir = dataset_root / "labels" / split
        for label_file in labels_dir.glob("*.txt"):
            text = label_file.read_text(encoding="utf-8").strip()
            if not text:
                continue

            new_lines: list[str] = []
            for line in text.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                old_idx = int(parts[0])
                if old_idx not in index_remap:
                    continue
                new_idx = index_remap[old_idx]
                new_lines.append(" ".join([str(new_idx), *parts[1:]]))

            label_file.write_text("\n".join(new_lines), encoding="utf-8")

    yaml_data = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "names": [name for _, name in sorted_items],
    }

    output_yaml = dataset_root / "dataset.yaml"
    output_yaml.write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")
    return output_yaml


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    root = args.project_root
    raw_images_dir = root / "data" / "raw" / "images"
    raw_videos_dir = root / "data" / "raw" / "videos"
    archive_dir = root / "data" / "yolo_dataset" / "archive"

    extracted_dir = root / "data" / "extracted_frames"
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = root / "data" / "yolo_dataset"
    train_images = dataset_root / "images" / "train"
    val_images = dataset_root / "images" / "val"
    train_labels = dataset_root / "labels" / "train"
    val_labels = dataset_root / "labels" / "val"

    # Only reset generated outputs, never remove the whole dataset root.
    for generated_path in [dataset_root / "images", dataset_root / "labels", dataset_root / "dataset.yaml"]:
        if generated_path.is_dir():
            shutil.rmtree(generated_path)
        elif generated_path.is_file():
            generated_path.unlink()

    for folder in [train_images, val_images, train_labels, val_labels]:
        folder.mkdir(parents=True, exist_ok=True)

    raw_images = find_media_files(raw_images_dir, IMAGE_EXTENSIONS) if raw_images_dir.exists() else []
    raw_videos = find_media_files(raw_videos_dir, VIDEO_EXTENSIONS) if raw_videos_dir.exists() else []

    external_images = find_media_files(args.datasets_root, IMAGE_EXTENSIONS) if args.datasets_root.exists() else []
    external_videos = find_media_files(args.datasets_root, VIDEO_EXTENSIONS) if args.datasets_root.exists() else []
    if external_images or external_videos:
        print(
            "[INFO] Found external datasets media "
            f"(images={len(external_images)}, videos={len(external_videos)}) at {args.datasets_root}"
        )
    raw_images.extend(external_images)
    raw_videos.extend(external_videos)

    archive_images = find_media_files(archive_dir, IMAGE_EXTENSIONS) if archive_dir.exists() else []
    archive_videos = find_media_files(archive_dir, VIDEO_EXTENSIONS) if archive_dir.exists() else []

    if archive_images or archive_videos:
        print(
            "[INFO] Found archive media "
            f"(images={len(archive_images)}, videos={len(archive_videos)}) at {archive_dir}"
        )

    raw_images.extend(archive_images)
    raw_videos.extend(archive_videos)

    if args.max_videos > 0 and len(raw_videos) > args.max_videos:
        print(f"[INFO] Limiting videos to first {args.max_videos} out of {len(raw_videos)}")
        raw_videos = raw_videos[: args.max_videos]

    extracted_frames: list[Path] = []
    for video in raw_videos:
        extracted_frames.extend(
            extract_frames(
                video,
                extracted_dir,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames_per_video,
            )
        )

    all_images = raw_images + extracted_frames
    if not all_images:
        raise RuntimeError(
            "No media found. Put files in data/raw/images and/or data/raw/videos first."
        )

    random.shuffle(all_images)
    split_index = int((1.0 - args.val_ratio) * len(all_images))
    train_sources = all_images[:split_index]
    val_sources = all_images[split_index:]

    if not train_sources or not val_sources:
        raise RuntimeError("Train/val split is empty. Add more media or adjust --val-ratio.")

    all_target_images: list[Path] = []
    all_target_labels: list[Path] = []

    skipped_unreadable = 0

    for split_name, sources, split_images, split_labels in [
        ("train", train_sources, train_images, train_labels),
        ("val", val_sources, val_images, val_labels),
    ]:
        for idx, source in enumerate(sources):
            if not is_readable_image(source):
                skipped_unreadable += 1
                continue

            output_image = split_images / f"{split_name}_{idx:06d}.jpg"
            output_label = split_labels / f"{split_name}_{idx:06d}.txt"

            shutil.copy2(source, output_image)
            copy_existing_label_if_present(source, output_label)

            all_target_images.append(output_image)
            all_target_labels.append(output_label)

    class_names_detected: dict[int, str] = {}

    if args.auto_label:
        classes_to_keep = {name.strip().lower() for name in args.classes.split(",") if name.strip()}
        discovered_dataset_classes = discover_dataset_classes(args.datasets_root)
        if discovered_dataset_classes:
            classes_to_keep.update(discovered_dataset_classes)
            print(f"[INFO] Added {len(discovered_dataset_classes)} classes discovered from datasets folder.")
        class_names_detected, labeled_count = auto_label_images(
            all_target_images,
            all_target_labels,
            model_name=args.auto_label_model,
            classes_to_keep=classes_to_keep,
            conf=args.auto_label_conf,
            batch_size=max(1, args.auto_label_batch_size),
        )
        print(f"[INFO] Auto-labeled images with detections: {labeled_count}/{len(all_target_images)}")

    if not class_names_detected:
        for label_file in all_target_labels:
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 5:
                    cls_idx = int(parts[0])
                    class_names_detected.setdefault(cls_idx, f"class_{cls_idx}")

    dataset_yaml = write_dataset_yaml(dataset_root, class_names_detected)

    print("[DONE] Dataset prepared")
    print(f"[DONE] Skipped unreadable images: {skipped_unreadable}")
    print(f"[DONE] Total images: {len(all_target_images)}")
    print(f"[DONE] Train images: {len(train_sources)} | Val images: {len(val_sources)}")
    print(f"[DONE] Dataset config: {dataset_yaml}")


if __name__ == "__main__":
    main()
