from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from ultralytics import YOLO


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
TARGET_OBJECTS = {
    "person",
    "handbag",
    "backpack",
    "suitcase",
    "knife",
    "scissors",
    "baseball bat",
    "gun",
    "pistol",
    "rifle",
}

DEFAULT_DATASETS_ROOT = Path(__file__).resolve().parent.parent / "datasets"
WEAPON_KEYWORDS = {"knife", "gun", "pistol", "rifle", "bat", "scissors", "weapon"}
BAG_KEYWORDS = {"bag", "handbag", "backpack", "suitcase", "purse", "luggage"}
PERSON_KEYWORDS = {"person", "people", "man", "woman", "child", "human", "pedestrian"}


@dataclass
class TrackState:
    frames_seen: int = 0
    stationary_frames: int = 0
    last_center: tuple[float, float] | None = None


def normalize_label(value: str) -> str:
    normalized = "_".join(str(value or "").strip().lower().replace("-", " ").split())
    return normalized


@lru_cache(maxsize=4)
def discover_dataset_context(datasets_root: str) -> dict[str, set[str]]:
    root = Path(datasets_root)
    object_labels: set[str] = set()
    interaction_labels: set[str] = set()

    if not root.exists():
        return {"objects": object_labels, "interactions": interaction_labels}

    for csv_path in root.rglob("*.csv"):
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                lower_fields = {name.lower(): name for name in reader.fieldnames}
                class_field = next(
                    (
                        lower_fields[key]
                        for key in ("class", "label", "name", "category")
                        if key in lower_fields
                    ),
                    None,
                )
                if class_field is None:
                    continue
                for row in reader:
                    raw = row.get(class_field, "")
                    label = normalize_label(raw)
                    if label:
                        object_labels.add(label)
        except (OSError, UnicodeDecodeError, csv.Error):
            continue

    for xml_path in root.rglob("*.xml"):
        try:
            xml_root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError):
            continue

        for elem in xml_root.findall(".//label"):
            name_attr = normalize_label(elem.attrib.get("label", ""))
            if name_attr:
                object_labels.add(name_attr)

            if "name" in {child.tag for child in list(elem)}:
                name_node = elem.find("name")
                if name_node is not None and name_node.text:
                    text_label = normalize_label(name_node.text)
                    if text_label:
                        object_labels.add(text_label)

        for elem in xml_root.findall(".//*[@label]"):
            label_attr = normalize_label(elem.attrib.get("label", ""))
            if label_attr:
                object_labels.add(label_attr)

    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue
        label = normalize_label(folder.name)
        if not label:
            continue

        if any(token in label for token in ("violence", "fight", "assault", "weapon", "crime")):
            interaction_labels.add(label)

    return {"objects": object_labels, "interactions": interaction_labels}


def infer_interactions(
    detections: list[dict[str, Any]],
    dataset_context: dict[str, set[str]],
) -> list[str]:
    labels = [normalize_label(str(item.get("class_name", ""))) for item in detections]
    labels = [label for label in labels if label]

    if not labels:
        return []

    person_count = sum(
        1 for label in labels if label in dataset_context["objects"] and any(token in label for token in PERSON_KEYWORDS)
    )
    if person_count == 0:
        person_count = sum(1 for label in labels if any(token in label for token in PERSON_KEYWORDS))

    bag_count = sum(1 for label in labels if any(token in label for token in BAG_KEYWORDS))
    weapon_count = sum(1 for label in labels if any(token in label for token in WEAPON_KEYWORDS))

    interactions: list[str] = []
    if person_count >= 2 and weapon_count > 0:
        interactions.append("armed_group_encounter")
    if person_count >= 1 and weapon_count > 0:
        interactions.append("person_with_weapon")
    if person_count >= 1 and bag_count > 0:
        interactions.append("person_carrying_bag")
    if person_count >= 3:
        interactions.append("crowd_gathering")
    if weapon_count > 0 and person_count == 0:
        interactions.append("weapon_without_visible_person")

    has_violence_dataset = any("violence" in value for value in dataset_context["interactions"])
    has_nonviolence_dataset = any("nonviolence" in value for value in dataset_context["interactions"])
    if has_violence_dataset and (weapon_count > 0 or person_count >= 2):
        interactions.append("possible_violence_pattern")
    elif has_nonviolence_dataset and weapon_count == 0:
        interactions.append("non_violent_scene_pattern")

    deduped = sorted(set(interactions))
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO tracking and categorize activity.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Video file or folder of videos.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "crime_detector" / "weights" / "best.pt",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path(__file__).resolve().parent / "activity_rules.json",
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--stationary-threshold", type=float, default=0.012)
    parser.add_argument("--save-annotated", action="store_true")
    return parser.parse_args()


def list_video_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        return [path]
    if path.is_dir():
        return sorted(
            [
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
            ]
        )
    return []


def load_rules(rules_path: Path) -> dict[str, dict[str, float]]:
    if not rules_path.exists():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")
    return json.loads(rules_path.read_text(encoding="utf-8"))


def load_model(model_path: Path) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    return YOLO(str(model_path))


def summarize_detections(detections: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for item in detections:
        name = str(item.get("class_name", "object")).lower()
        summary[name] = summary.get(name, 0) + 1
    return dict(sorted(summary.items(), key=lambda kv: (-kv[1], kv[0])))


def apply_person_identity_labels(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    people = [d for d in detections if d.get("class_name") == "person" and len(d.get("box", [])) >= 4]
    people_sorted = sorted(people, key=lambda d: float(d["box"][0]))
    person_ids: dict[int, str] = {id(det): f"person_{idx + 1}" for idx, det in enumerate(people_sorted)}

    for det in detections:
        cls = str(det.get("class_name", "object"))
        if cls == "person":
            det["display_name"] = person_ids.get(id(det), "person")
        else:
            det["display_name"] = cls
    return detections


def extract_detections_from_result(result: Any, model: YOLO) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes
    cls_list = boxes.cls.cpu().tolist()
    xyxy_list = boxes.xyxy.cpu().tolist()
    conf_list = boxes.conf.cpu().tolist() if boxes.conf is not None else [0.0] * len(cls_list)

    for cls_raw, box, det_conf in zip(cls_list, xyxy_list, conf_list):
        cls_idx = int(cls_raw)
        class_name = str(model.names.get(cls_idx, cls_idx)).lower()
        detections.append(
            {
                "class_id": cls_idx,
                "class_name": class_name,
                "confidence": float(det_conf),
                "box": [float(value) for value in box],
            }
        )

    return detections


def classify_activity(metrics: dict[str, float], rules: dict[str, dict[str, float]]) -> dict[str, Any]:
    if (
        metrics["weapon_frame_ratio"] >= rules["aggressive_confrontation"]["weapon_frame_ratio"]
        and metrics["avg_people"] >= rules["aggressive_confrontation"]["min_people"]
    ):
        return {
            "activity": "aggressive_confrontation",
            "confidence": min(0.99, 0.6 + metrics["weapon_frame_ratio"] * 1.5),
        }

    if (
        metrics["avg_people"] >= rules["group_tension"]["min_avg_people"]
        and metrics["motion_score"] >= rules["group_tension"]["min_motion_score"]
    ):
        return {
            "activity": "group_tension",
            "confidence": min(0.95, 0.52 + metrics["motion_score"] * 8.0),
        }

    if (
        metrics["motion_score"] >= rules["rapid_running"]["min_motion_score"]
        and metrics["avg_people"] >= rules["rapid_running"]["min_avg_people"]
    ):
        return {
            "activity": "rapid_running",
            "confidence": min(0.93, 0.5 + metrics["motion_score"] * 7.0),
        }

    if (
        metrics["loitering_ratio"] >= rules["suspicious_loitering"]["min_loitering_ratio"]
        and metrics["motion_score"] <= rules["suspicious_loitering"]["max_motion_score"]
    ):
        return {
            "activity": "suspicious_loitering",
            "confidence": min(0.9, 0.52 + metrics["loitering_ratio"] * 0.7),
        }

    return {
        "activity": "routine_activity",
        "confidence": max(0.5, 0.86 - metrics["motion_score"] * 6.0),
    }


def classify_from_detections(detections: list[dict[str, Any]], rules: dict[str, dict[str, float]]) -> dict[str, Any]:
    dataset_context = discover_dataset_context(str(DEFAULT_DATASETS_ROOT))
    people = sum(1 for item in detections if "person" in normalize_label(str(item.get("class_name", ""))))
    weapon_hits = sum(
        1
        for item in detections
        if any(token in normalize_label(str(item.get("class_name", ""))) for token in WEAPON_KEYWORDS)
    )
    total = max(1, len(detections))

    metrics = {
        "avg_people": float(people),
        "motion_score": 0.0,
        "weapon_frame_ratio": float(weapon_hits / total),
        "loitering_ratio": 0.0,
    }

    result = classify_activity(metrics, rules)
    if weapon_hits and people:
        result["activity"] = "aggressive_confrontation"
        result["confidence"] = min(0.95, 0.6 + weapon_hits / total)
    elif people >= 3:
        result["activity"] = "group_tension"
        result["confidence"] = min(0.9, 0.55 + people / 10.0)
    elif people <= 1:
        result["activity"] = "routine_activity"
        result["confidence"] = max(0.5, result["confidence"])

    result["metrics"] = metrics
    result["interactions"] = infer_interactions(detections, dataset_context)
    return result


def analyze_image(
    image_path: Path,
    model: YOLO,
    rules: dict[str, dict[str, float]],
    conf: float,
    object_model: YOLO | None = None,
) -> dict[str, Any]:
    detector = object_model if object_model is not None else model
    detection_conf = max(0.01, min(0.99, float(conf)))
    primary_results = detector.predict(source=str(image_path), conf=detection_conf, verbose=False)
    detections = extract_detections_from_result(primary_results[0], detector)

    detections = apply_person_identity_labels(detections)

    activity_result = classify_from_detections(detections, rules)
    dataset_context = discover_dataset_context(str(DEFAULT_DATASETS_ROOT))
    shape_source = primary_results[0]
    image_height, image_width = shape_source.orig_shape[:2]
    return {
        "image": str(image_path),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "detections": detections,
        "object_summary": summarize_detections(detections),
        "interactions": infer_interactions(detections, dataset_context),
        "result": activity_result,
    }


def analyze_video(
    video_path: Path,
    model: YOLO,
    rules: dict[str, dict[str, float]],
    conf: float,
    stationary_threshold: float,
    output_dir: Path,
    save_annotated: bool,
    object_model: YOLO | None = None,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not read video: {video_path}")

    frame_count = 0
    people_counts: list[int] = []
    motion_samples: list[float] = []
    weapon_frames = 0
    sample_stride = 12
    sampled_frames: list[np.ndarray] = []
    last_sample_brightness: float | None = None

    weapon_keywords = {"knife", "gun", "pistol", "rifle", "baseball bat", "scissors"}
    object_counts: dict[str, int] = {}
    max_people_in_frame = 0
    preview_detections: list[dict[str, Any]] = []
    preview_frame_rank = -1.0
    preview_width = 0
    preview_height = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        if frame_count % sample_stride != 0:
            continue

        detector = object_model if object_model is not None else model
        detection_conf = max(0.01, min(0.99, float(conf)))
        results = detector.predict(source=frame, conf=detection_conf, verbose=False)
        result = results[0]
        frame_detections = extract_detections_from_result(result, detector)

        sample_people = 0
        sample_weapon = False

        for item in frame_detections:
            class_name = str(item.get("class_name", "")).lower()
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            if class_name == "person":
                sample_people += 1
            if any(keyword in class_name for keyword in weapon_keywords):
                sample_weapon = True

        frame_rank = sample_people * 10.0 + (5.0 if sample_weapon else 0.0) + len(frame_detections) * 0.2
        if frame_rank > preview_frame_rank and frame_detections:
            preview_frame_rank = frame_rank
            preview_detections = apply_person_identity_labels(frame_detections.copy())[:20]
            preview_height, preview_width = result.orig_shape[:2]

        max_people_in_frame = max(max_people_in_frame, sample_people)

        people_counts.append(sample_people)
        if sample_weapon:
            weapon_frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        if last_sample_brightness is None:
            motion_samples.append(0.0)
        else:
            motion_samples.append(abs(brightness - last_sample_brightness) / 255.0)
        last_sample_brightness = brightness
        sampled_frames.append(frame)

    cap.release()

    if frame_count == 0 or not sampled_frames:
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    avg_people = float(np.mean(people_counts)) if people_counts else 0.0
    motion_score = float(np.mean(motion_samples)) if motion_samples else 0.0
    weapon_frame_ratio = weapon_frames / frame_count

    loitering_ratio = max(0.0, min(1.0, 1.0 - motion_score * 18.0))

    metrics = {
        "avg_people": avg_people,
        "motion_score": motion_score,
        "weapon_frame_ratio": weapon_frame_ratio,
        "loitering_ratio": loitering_ratio,
        "frames": float(frame_count),
    }

    activity_result = classify_activity(metrics, rules)
    dataset_context = discover_dataset_context(str(DEFAULT_DATASETS_ROOT))

    if save_annotated:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / f"{video_path.stem}_summary.json"
        summary_path.write_text(json.dumps({"metrics": metrics, "result": activity_result}, indent=2), encoding="utf-8")

    return {
        "video": str(video_path),
        "metrics": metrics,
        "object_summary": dict(sorted(object_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "max_people_in_frame": int(max_people_in_frame),
        "preview_detections": preview_detections,
        "interactions": infer_interactions(preview_detections, dataset_context),
        "image_width": int(preview_width),
        "image_height": int(preview_height),
        "result": activity_result,
    }


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"YOLO model not found: {args.model}")

    videos = list_video_files(args.input)
    if not videos:
        raise RuntimeError("No videos found. Provide a valid video file or folder.")

    rules = load_rules(args.rules)
    model = YOLO(str(args.model))
    object_model = None
    object_model_path = args.project_root / "yolov8n.pt"
    if object_model_path.exists():
        object_model = YOLO(str(object_model_path))

    output_dir = args.project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    for video_path in videos:
        print(f"[INFO] Analyzing {video_path.name}...")
        result = analyze_video(
            video_path=video_path,
            model=model,
            rules=rules,
            conf=args.conf,
            stationary_threshold=args.stationary_threshold,
            output_dir=output_dir,
            save_annotated=args.save_annotated,
            object_model=object_model,
        )
        all_results.append(result)

        activity = result["result"]["activity"]
        confidence = float(result["result"]["confidence"]) * 100.0
        print(f"[DONE] {video_path.name}: {activity} ({confidence:.1f}%)")

    summary_path = output_dir / "activity_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"[DONE] Saved activity summary: {summary_path}")


if __name__ == "__main__":
    main()
