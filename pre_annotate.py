"""
pre_annotate.py
---------------
Auto-classifies each human box in the test label files into:
  class 3  →  rider_with_helmet
  class 4  →  rider_without_helmet
  class 1  →  human (non-rider / pedestrian, left unchanged)

Uses the same spatial scoring as helmet_logic.py so the pre-annotation
reflects the rule-based system's own logic. The reviewer only needs to
correct the cases where the auto-guess is wrong.

Output: test/labels_annotated/  (original files are not overwritten)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2

ROOT        = Path(__file__).parent
SRC_LABELS  = ROOT / "test" / "labels"
SRC_IMAGES  = ROOT / "test" / "images"
DST_LABELS  = ROOT / "rules_evaluation" / "labels_annotated"

# Class IDs in the output
CLS_HELMET      = 0
CLS_HUMAN       = 1   # non-rider
CLS_MOTORCYCLE  = 2
CLS_RIDER_HELM  = 3   # rider with helmet
CLS_RIDER_NOH   = 4   # rider without helmet

CLASS_NAMES = ["helmet", "human", "motorcycle"]

sys.path.insert(0, str(ROOT))
from helmet_logic import (
    _rider_score, _head_region, _helmet_score,
    RIDER_SCORE_THRESHOLD, HELMET_SCORE_THRESHOLD,
)


def yolo_to_pixel(cx, cy, bw, bh, img_w, img_h):
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h
    return [x1, y1, x2, y2]


def pixel_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def parse_label_row(line: str):
    """Return a YOLO box row, converting segmentation polygons when needed."""
    parts = line.split()
    if len(parts) == 5:
        return (int(parts[0]), float(parts[1]), float(parts[2]),
                float(parts[3]), float(parts[4]))

    # YOLO segmentation rows contain: class x1 y1 x2 y2 ... xn yn.
    if len(parts) >= 7 and len(parts) % 2 == 1:
        cls = int(parts[0])
        coords = [float(value) for value in parts[1:]]
        xs = []
        ys = []
        for index, value in enumerate(coords):
            if index % 2 == 0:
                xs.append(value)
            else:
                ys.append(value)
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return (cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)

    return None


def get_image_size(img_path: Path):
    image = cv2.imread(str(img_path))
    if image is None:
        return 640, 640

    img_h, img_w = image.shape[:2]
    return img_w, img_h


def pre_annotate_file(label_path: Path, img_path: Path) -> list[str]:
    """Return new label lines with humans reclassified."""
    img_w, img_h = get_image_size(img_path)

    raw_lines = label_path.read_text().splitlines()
    rows = []
    for line in raw_lines:
        row = parse_label_row(line)
        if row is not None:
            rows.append(row)

    helmets = []
    humans = []
    motorcycles = []
    for cls, cx, cy, bw, bh in rows:
        pixel_box = yolo_to_pixel(cx, cy, bw, bh, img_w, img_h)
        if cls == CLS_HELMET:
            helmets.append(pixel_box)
        elif cls == CLS_HUMAN:
            yolo_coords = (cx, cy, bw, bh)
            humans.append((pixel_box, yolo_coords))
        elif cls == CLS_MOTORCYCLE:
            motorcycles.append(pixel_box)

    new_classes: dict[tuple, int] = {}  # yolo coords → new class id

    for box_px, yolo_coords in humans:
        # Find best motorcycle match
        best_rscore = 0.0
        for motorcycle_box in motorcycles:
            score = _rider_score(box_px, motorcycle_box)
            if score > best_rscore:
                best_rscore = score

        if best_rscore < RIDER_SCORE_THRESHOLD:
            new_classes[yolo_coords] = CLS_HUMAN
            continue

        # It's a rider — check for helmet
        head = _head_region(box_px)
        best_hscore = 0.0
        for helmet_box in helmets:
            score = _helmet_score(head, helmet_box)
            if score > best_hscore:
                best_hscore = score

        if best_hscore >= HELMET_SCORE_THRESHOLD:
            new_classes[yolo_coords] = CLS_RIDER_HELM
        else:
            new_classes[yolo_coords] = CLS_RIDER_NOH

    # Build output lines
    out_lines = []
    for cls, cx, cy, bw, bh in rows:
        if cls == CLS_HUMAN:
            new_cls = new_classes.get((cx, cy, bw, bh), CLS_HUMAN)
            out_lines.append(f"{new_cls} {cx} {cy} {bw} {bh}")
        else:
            out_lines.append(f"{cls} {cx} {cy} {bw} {bh}")
    return out_lines


def main():
    DST_LABELS.mkdir(parents=True, exist_ok=True)

    label_files = sorted(SRC_LABELS.glob("*.txt"))
    counts = {3: 0, 4: 0, 1: 0}

    for lf in label_files:
        stem = lf.stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            possible_path = SRC_IMAGES / (stem + ext)
            if possible_path.exists():
                img_path = possible_path
                break

        if img_path is None:
            shutil.copy(lf, DST_LABELS / lf.name)
            continue

        new_lines = pre_annotate_file(lf, img_path)
        (DST_LABELS / lf.name).write_text("\n".join(new_lines) + "\n")

        for line in new_lines:
            c = int(line.split()[0])
            if c in counts:
                counts[c] += 1

    print(f"Pre-annotation complete: {DST_LABELS}")
    print(f"  {len(label_files)} files processed")
    print(f"  Humans classified:")
    print(f"    class 3  rider + helmet        : {counts[3]}")
    print(f"    class 4  rider + no helmet     : {counts[4]}")
    print(f"    class 1  non-rider (pedestrian): {counts[1]}")
    print(f"\nNext step: run  python review_annotations.py")


if __name__ == "__main__":
    main()
