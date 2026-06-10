"""
generate_report_images.py
-------------------------
Generates report-ready images for the rules evaluation set using the
yolov8n_e60 model.

Output
------
  rules_evaluation/report_images/
    comparisons/
      testhelmet001.jpg
      testhelmet002.jpg
      ...
    predictions/
      testhelmet001.jpg
      testhelmet002.jpg
      ...

Each comparison image is a two-panel layout:
  LEFT  - Ground truth from rules_evaluation/labels_annotated/
          (5-class compliance labels drawn with colour coding)
  RIGHT - Full pipeline output (YOLO -> analyze_detections)
          showing riders with green (helmet) or red (no helmet) boxes
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from helmet_logic import analyze_detections, draw_analysis

# -- Paths -----------------------------------------------------------------
TEST_IMAGES_DIR = ROOT / "test" / "images"
GT_LABELS_DIR = ROOT / "rules_evaluation" / "labels_annotated"
MODEL_PATH = ROOT / "exported_models" / "helmet_data_yolov8n_e60_best.pt"
OUTPUT_DIR = ROOT / "rules_evaluation" / "report_images"
COMPARISONS_DIR = OUTPUT_DIR / "comparisons"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

CLASS_NAMES = ["helmet", "human", "motorcycle"]
CONF_THRESHOLD = 0.35
MATCH_IOU = 0.30

# -- GT drawing colours (BGR) ---------------------------------------------
GT_COLORS = {
    0: (180, 40, 200),   # helmet - purple
    1: (40, 160, 235),   # pedestrian - orange
    2: (240, 130, 60),   # motorcycle - blue
    3: (30, 200, 30),    # rider + helmet - green
    4: (30, 30, 220),    # rider, no helmet - red
}
GT_LABEL_TEXT = {
    0: "GT: Helmet",
    1: "GT: Pedestrian",
    2: "GT: Motorcycle",
    3: "GT: Rider + Helmet",
    4: "GT: Rider - No Helmet",
}


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union else 0.0


def load_gt_boxes(label_path: Path, img_w: int, img_h: int):
    """Return list of (cls, x1, y1, x2, y2) in pixel coords."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = (float(p) for p in parts[1:])
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def load_gt_riders(gt_boxes):
    return [
        {"box": [x1, y1, x2, y2], "has_helmet": cls == 3}
        for cls, x1, y1, x2, y2 in gt_boxes
        if cls in (3, 4)
    ]


def _label(canvas, text, x1, y1, color):
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    lx = max(0, min(int(x1), canvas.shape[1] - tw - 8))
    ly = max(th + 4, int(y1) - 4)
    cv2.rectangle(canvas, (lx, ly - th - 4), (lx + tw + 6, ly + 2), color, -1)
    cv2.putText(canvas, text, (lx + 3, ly - 2), font, scale, (255, 255, 255), thick)


def draw_gt_panel(image: np.ndarray, gt_boxes) -> np.ndarray:
    """Draw ground-truth compliance boxes."""
    canvas = image.copy()
    for priority in (False, True):
        for cls, x1, y1, x2, y2 in gt_boxes:
            is_rider = cls in (3, 4)
            if is_rider != priority:
                continue
            color = GT_COLORS.get(cls, (180, 180, 180))
            thick = 3 if is_rider else 2
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
            _label(canvas, GT_LABEL_TEXT.get(cls, str(cls)), x1, y1, color)
    return canvas


def compute_stats(gt_riders, pred_riders) -> dict:
    pairs = sorted(
        [
            (_iou(gr["box"], pr["human"]["box"]), gi, pi)
            for gi, gr in enumerate(gt_riders)
            for pi, pr in enumerate(pred_riders)
        ],
        reverse=True,
    )
    matched_gt, matched_pred, matches = set(), set(), []
    for iou, gi, pi in pairs:
        if iou < MATCH_IOU or gi in matched_gt or pi in matched_pred:
            continue
        matches.append((gi, pi))
        matched_gt.add(gi)
        matched_pred.add(pi)

    fn = len(gt_riders) - len(matched_gt)
    fp = len(pred_riders) - len(matched_pred)
    false_safe = false_noh = 0
    for gi, pi in matches:
        gt_has_helmet = gt_riders[gi]["has_helmet"]
        pred_has_helmet = pred_riders[pi]["wearing_helmet"]
        if not gt_has_helmet and pred_has_helmet:
            false_safe += 1
        if gt_has_helmet and not pred_has_helmet:
            false_noh += 1

    return {
        "n_gt": len(gt_riders),
        "n_pred": len(pred_riders),
        "rider_fn": fn,
        "rider_fp": fp,
        "false_safe": false_safe,
        "false_noh": false_noh,
    }


def make_comparison(
    gt_img: np.ndarray,
    pred_img: np.ndarray,
    img_name: str,
    stats: dict,
) -> np.ndarray:
    """Stack GT and prediction side by side with a stats footer."""
    h = gt_img.shape[0]
    w = gt_img.shape[1]

    if pred_img.shape[:2] != (h, w):
        pred_img = cv2.resize(pred_img, (w, h))

    header_h = 36
    footer_h = 28
    canvas = np.ones((h + header_h + footer_h, w * 2, 3), dtype=np.uint8) * 245

    canvas[header_h : header_h + h, :w] = gt_img
    canvas[header_h : header_h + h, w:] = pred_img
    cv2.line(canvas, (w, 0), (w, h + header_h + footer_h), (160, 160, 160), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Ground Truth", (10, 24), font, 0.68, (30, 30, 30), 2)
    cv2.putText(canvas, "System Output", (w + 10, 24), font, 0.68, (30, 30, 30), 2)

    (name_w, _), _ = cv2.getTextSize(img_name, font, 0.48, 1)
    cv2.putText(canvas, img_name, (w * 2 - name_w - 8, 24), font, 0.48, (100, 100, 100), 1)

    footer_color = (220, 220, 220)
    if stats["false_safe"] > 0:
        footer_color = (180, 180, 255)
    elif stats["false_noh"] > 0 or stats["rider_fn"] > 0:
        footer_color = (200, 235, 255)
    elif stats["rider_fp"] > 0:
        footer_color = (220, 255, 220)

    cv2.rectangle(
        canvas,
        (0, header_h + h),
        (w * 2, header_h + h + footer_h),
        footer_color,
        -1,
    )

    footer_text = (
        f"  GT riders={stats['n_gt']}  |  "
        f"Predicted={stats['n_pred']}  |  "
        f"Missed riders (FN)={stats['rider_fn']}  |  "
        f"False riders (FP)={stats['rider_fp']}  |  "
        f"False safe={stats['false_safe']}  |  "
        f"False no-helmet={stats['false_noh']}"
    )
    cv2.putText(
        canvas,
        footer_text,
        (8, header_h + h + footer_h - 8),
        font,
        0.44,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )

    return canvas


def main():
    from ultralytics import YOLO

    COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model      : {MODEL_PATH.name}")
    print(f"GT labels  : {GT_LABELS_DIR}")
    print(f"Test images: {TEST_IMAGES_DIR}")
    print(f"Comparisons: {COMPARISONS_DIR}")
    print(f"Predictions: {PREDICTIONS_DIR}")
    print()

    model = YOLO(str(MODEL_PATH))

    image_files = sorted(
        f for f in TEST_IMAGES_DIR.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    total = len(image_files)
    saved_comparisons = 0
    saved_predictions = 0
    error_summary = {
        "false_safe": 0,
        "false_noh": 0,
        "missed_rider": 0,
        "false_rider": 0,
    }

    for i, img_path in enumerate(image_files, 1):
        label_path = GT_LABELS_DIR / (img_path.stem + ".txt")

        result = model(str(img_path), verbose=False, conf=CONF_THRESHOLD)[0]
        img_h, img_w = result.orig_shape

        pred_dets = []
        if result.boxes is not None:
            for box in result.boxes:
                pred_dets.append(
                    {
                        "class_name": CLASS_NAMES[int(box.cls[0])],
                        "box": box.xyxy[0].tolist(),
                        "confidence": float(box.conf[0]),
                    }
                )

        analysis = analyze_detections(pred_dets)
        gt_boxes = load_gt_boxes(label_path, img_w, img_h)
        gt_riders = load_gt_riders(gt_boxes)
        stats = compute_stats(gt_riders, analysis["riders"])

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [SKIP] could not read {img_path.name}")
            continue

        gt_panel = draw_gt_panel(img_bgr, gt_boxes)
        pred_panel = draw_analysis(img_bgr, analysis, show_labels=True)
        comparison = make_comparison(gt_panel, pred_panel, img_path.name, stats)

        pred_out_path = PREDICTIONS_DIR / img_path.name
        comp_out_path = COMPARISONS_DIR / img_path.name
        if cv2.imwrite(str(pred_out_path), pred_panel):
            saved_predictions += 1
        if cv2.imwrite(str(comp_out_path), comparison):
            saved_comparisons += 1

        if stats["false_safe"] > 0:
            error_summary["false_safe"] += 1
        if stats["false_noh"] > 0:
            error_summary["false_noh"] += 1
        if stats["rider_fn"] > 0:
            error_summary["missed_rider"] += 1
        if stats["rider_fp"] > 0:
            error_summary["false_rider"] += 1

        if i % 20 == 0 or i == total:
            print(
                f"  [{i:3d}/{total}] saved "
                f"{saved_comparisons} comparisons, {saved_predictions} predictions"
            )

    print(f"\nDone - saved {saved_comparisons}/{total} comparisons to:")
    print(f"  {COMPARISONS_DIR}")
    print(f"Done - saved {saved_predictions}/{total} predictions to:")
    print(f"  {PREDICTIONS_DIR}")
    print()
    print("Error summary (number of images containing each error type):")
    print(f"  False safe      (GT=no helmet, pred=helmet) : {error_summary['false_safe']:3d}")
    print(f"  False no-helmet (GT=helmet,    pred=none  ) : {error_summary['false_noh']:3d}")
    print(f"  Missed rider    (FN in rider detection    ) : {error_summary['missed_rider']:3d}")
    print(f"  False rider     (FP in rider detection    ) : {error_summary['false_rider']:3d}")


if __name__ == "__main__":
    main()
