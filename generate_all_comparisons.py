"""
generate_all_comparisons.py
----------------------------
Generates a side-by-side compliance comparison image for every one of the
200 test images using the yolov8n_e60 model.

Output
------
  rules_evaluation/report_images/comparisons/
      testhelmet001.jpg
      testhelmet002.jpg
      ...  (one file per test image)

Each output image is a two-panel layout:
  LEFT  – Ground truth from rules_evaluation/labels_annotated/
           (5-class compliance labels drawn with colour coding)
  RIGHT – Full pipeline output  (YOLO  ->  analyze_detections)
           showing riders with green (helmet) or red (no helmet) boxes

A footer bar below each panel shows per-image stats:
  GT riders / predicted riders / missed (FN) / false riders (FP)
  / false safe / false no-helmet

Colour legend
-------------
  Ground truth side          System output side
  Green  = rider + helmet    Green  = rider classified as helmeted
  Red    = rider, no helmet  Red    = rider classified as not helmeted
  Purple = helmet box        Purple = helmet box
  Blue   = motorcycle        Blue   = motorcycle
  Orange = pedestrian        (pedestrian boxes not shown on right side)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from helmet_logic import analyze_detections, draw_analysis

# ── Paths ──────────────────────────────────────────────────────────────────
TEST_IMAGES_DIR = ROOT / "test" / "images"
GT_LABELS_DIR   = ROOT / "rules_evaluation" / "labels_annotated"
MODEL_PATH      = ROOT / "exported_models" / "helmet_data_yolov8n_e60_best.pt"
OUTPUT_DIR      = ROOT / "rules_evaluation" / "report_images" / "comparisons"

CLASS_NAMES    = ["helmet", "human", "motorcycle"]
CONF_THRESHOLD = 0.35
MATCH_IOU      = 0.30

# ── GT drawing colours (BGR) ───────────────────────────────────────────────
GT_COLORS = {
    0: (180,  40, 200),   # helmet      – purple
    1: ( 40, 160, 235),   # pedestrian  – orange
    2: (240, 130,  60),   # motorcycle  – blue
    3: ( 30, 200,  30),   # rider+helm  – green  (thick)
    4: ( 30,  30, 220),   # rider-noh   – red    (thick)
}
GT_LABEL_TEXT = {
    0: "GT: Helmet",
    1: "GT: Pedestrian",
    2: "GT: Motorcycle",
    3: "GT: Rider + Helmet",
    4: "GT: Rider - No Helmet",
}


# ── Geometry ────────────────────────────────────────────────────────────────

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union else 0.0


# ── Label loading ───────────────────────────────────────────────────────────

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
        for cls, x1, y1, x2, y2 in gt_boxes if cls in (3, 4)
    ]


# ── Drawing helpers ─────────────────────────────────────────────────────────

def _label(canvas, text, x1, y1, color):
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    lx = max(0, min(int(x1), canvas.shape[1] - tw - 8))
    ly = max(th + 4, int(y1) - 4)
    cv2.rectangle(canvas, (lx, ly - th - 4), (lx + tw + 6, ly + 2), color, -1)
    cv2.putText(canvas, text, (lx + 3, ly - 2), font, scale, (255, 255, 255), thick)


def draw_gt_panel(image: np.ndarray, gt_boxes) -> np.ndarray:
    """Draw ground-truth compliance boxes (non-riders first, riders on top)."""
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


# ── Stats ────────────────────────────────────────────────────────────────────

def compute_stats(gt_riders, pred_riders) -> dict:
    pairs = sorted(
        [(_iou(gr["box"], pr["human"]["box"]), gi, pi)
         for gi, gr in enumerate(gt_riders)
         for pi, pr in enumerate(pred_riders)],
        reverse=True,
    )
    mg, mp, matches = set(), set(), []
    for iou, gi, pi in pairs:
        if iou < MATCH_IOU or gi in mg or pi in mp:
            continue
        matches.append((gi, pi)); mg.add(gi); mp.add(pi)

    fn = len(gt_riders) - len(mg)
    fp = len(pred_riders) - len(mp)
    hfp = hfn = 0
    for gi, pi in matches:
        gt_h = gt_riders[gi]["has_helmet"]
        pr_h = pred_riders[pi]["wearing_helmet"]
        if not gt_h and pr_h: hfp += 1
        if gt_h and not pr_h: hfn += 1

    return {
        "n_gt": len(gt_riders), "n_pred": len(pred_riders),
        "rider_fn": fn, "rider_fp": fp,
        "false_safe": hfp, "false_noh": hfn,
    }


# ── Panel builder ────────────────────────────────────────────────────────────

def make_comparison(
    gt_img: np.ndarray,
    pred_img: np.ndarray,
    img_name: str,
    stats: dict,
) -> np.ndarray:
    """
    Stack GT (left) and prediction (right) side by side.
    A header bar labels each side; a footer bar shows per-image stats.
    """
    h = gt_img.shape[0]
    w = gt_img.shape[1]

    # Resize pred panel to match GT dimensions if needed
    if pred_img.shape[:2] != (h, w):
        pred_img = cv2.resize(pred_img, (w, h))

    HEADER_H = 36
    FOOTER_H = 28

    canvas = np.ones((h + HEADER_H + FOOTER_H, w * 2, 3), dtype=np.uint8) * 245

    # Paste panels
    canvas[HEADER_H : HEADER_H + h, :w]  = gt_img
    canvas[HEADER_H : HEADER_H + h, w:]  = pred_img

    # Divider
    cv2.line(canvas, (w, 0), (w, h + HEADER_H + FOOTER_H), (160, 160, 160), 2)

    # Header labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Ground Truth",  (10, 24), font, 0.68, (30, 30, 30), 2)
    cv2.putText(canvas, "System Output", (w + 10, 24), font, 0.68, (30, 30, 30), 2)

    # Image name (top-right of header)
    name_text = img_name
    (nw, _), _ = cv2.getTextSize(name_text, font, 0.48, 1)
    cv2.putText(canvas, name_text, (w * 2 - nw - 8, 24), font, 0.48, (100, 100, 100), 1)

    # Footer stats
    s = stats
    # Colour-code the footer based on error presence
    footer_color = (220, 220, 220)
    if s["false_safe"] > 0:
        footer_color = (180, 180, 255)   # red tint  – most dangerous error
    elif s["false_noh"] > 0 or s["rider_fn"] > 0:
        footer_color = (200, 235, 255)   # amber tint – notable errors
    elif s["rider_fp"] > 0:
        footer_color = (220, 255, 220)   # green tint – minor error

    cv2.rectangle(canvas,
                  (0, HEADER_H + h),
                  (w * 2, HEADER_H + h + FOOTER_H),
                  footer_color, -1)

    footer_text = (
        f"  GT riders={s['n_gt']}  |  "
        f"Predicted={s['n_pred']}  |  "
        f"Missed riders (FN)={s['rider_fn']}  |  "
        f"False riders (FP)={s['rider_fp']}  |  "
        f"False safe={s['false_safe']}  |  "
        f"False no-helmet={s['false_noh']}"
    )
    cv2.putText(canvas, footer_text,
                (8, HEADER_H + h + FOOTER_H - 8),
                font, 0.44, (40, 40, 40), 1, cv2.LINE_AA)

    return canvas


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    from ultralytics import YOLO

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model      : {MODEL_PATH.name}")
    print(f"GT labels  : {GT_LABELS_DIR}")
    print(f"Test images: {TEST_IMAGES_DIR}")
    print(f"Output     : {OUTPUT_DIR}")
    print()

    model = YOLO(str(MODEL_PATH))

    image_files = sorted(
        f for f in TEST_IMAGES_DIR.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    total = len(image_files)
    saved = 0
    error_summary = {"false_safe": 0, "false_noh": 0, "missed_rider": 0, "false_rider": 0}

    for i, img_path in enumerate(image_files, 1):
        label_path = GT_LABELS_DIR / (img_path.stem + ".txt")

        # YOLO inference
        result     = model(str(img_path), verbose=False, conf=CONF_THRESHOLD)[0]
        img_h, img_w = result.orig_shape

        pred_dets = []
        if result.boxes is not None:
            for box in result.boxes:
                pred_dets.append({
                    "class_name": CLASS_NAMES[int(box.cls[0])],
                    "box":        box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                })

        analysis  = analyze_detections(pred_dets)
        gt_boxes  = load_gt_boxes(label_path, img_w, img_h)
        gt_riders = load_gt_riders(gt_boxes)
        stats     = compute_stats(gt_riders, analysis["riders"])

        # Draw panels
        img_bgr  = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [SKIP] could not read {img_path.name}")
            continue

        gt_panel   = draw_gt_panel(img_bgr, gt_boxes)
        pred_panel = draw_analysis(img_bgr, analysis, show_labels=True)

        panel = make_comparison(gt_panel, pred_panel, img_path.name, stats)

        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), panel)
        saved += 1

        # Accumulate error summary
        if stats["false_safe"]  > 0: error_summary["false_safe"]    += 1
        if stats["false_noh"]   > 0: error_summary["false_noh"]     += 1
        if stats["rider_fn"]    > 0: error_summary["missed_rider"]  += 1
        if stats["rider_fp"]    > 0: error_summary["false_rider"]   += 1

        if i % 20 == 0 or i == total:
            print(f"  [{i:3d}/{total}]  saved {saved} images …")

    print(f"\nDone — {saved}/{total} comparison images saved to:")
    print(f"  {OUTPUT_DIR}")
    print()
    print("Error summary (number of images containing each error type):")
    print(f"  False safe      (GT=no helmet, pred=helmet) : {error_summary['false_safe']:3d}  <-- most dangerous")
    print(f"  False no-helmet (GT=helmet,    pred=none  ) : {error_summary['false_noh']:3d}")
    print(f"  Missed rider    (FN in rider detection    ) : {error_summary['missed_rider']:3d}")
    print(f"  False rider     (FP in rider detection    ) : {error_summary['false_rider']:3d}")
    print()
    print("Footer colour guide:")
    print("  Red tint    – image contains a false safe error")
    print("  Amber tint  – image has missed rider or false no-helmet")
    print("  Green tint  – image only has a false rider (FP)")
    print("  Grey        – no errors (correct prediction)")


if __name__ == "__main__":
    main()
