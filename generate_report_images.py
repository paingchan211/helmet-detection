"""
generate_report_images.py
--------------------------
Generates compliance comparison images for sections 3.7 and 3.8 using
the yolov8n_e60 model.

Each image is a side-by-side panel:
  LEFT  – Ground truth (from labels_annotated, 5-class compliance labels)
  RIGHT – System output (YOLO + rule-based logic via analyze_detections)

Images are organised into output folders by category:

  report_images/
    section_3_7_generalisation/
      good_single/        – single rider, all correct
      good_multi/         – multiple riders, mostly correct
      challenging_missed/ – riders in scene but system missed them (recall failure)
      challenging_crowd/  – busy scene with many objects
    section_3_8_errors/
      missed_rider/       – GT rider exists but system missed (FN)
      false_safe/         – GT no-helmet but system said helmet (dangerous)
      false_no_helmet/    – GT helmet but system said no-helmet (unfair)
      false_rider/        – GT pedestrian but system said rider (FP)
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
OUTPUT_DIR      = ROOT / "rules_evaluation" / "report_images"

CLASS_NAMES    = ["helmet", "human", "motorcycle"]
CONF_THRESHOLD = 0.35
MATCH_IOU      = 0.30

# How many example images to save per category (use None to save all found)
EXAMPLES_PER_CATEGORY = {
    # section 3.7
    "good_single":          3,
    "good_multi":           3,
    "challenging_missed":   3,
    "challenging_crowd":    3,
    # section 3.8  – save more so there's variety to pick from
    "missed_rider":         6,
    "false_safe":           None,   # only 3 exist, save all of them
    "false_no_helmet":      None,   # only 7 exist, save all of them
    "false_rider":          6,
}

# ── GT class colours (BGR) ─────────────────────────────────────────────────
GT_COLORS = {
    0: (180,  40, 200),   # helmet      – purple
    1: ( 40, 160, 235),   # pedestrian  – orange
    2: (240, 130,  60),   # motorcycle  – blue
    3: ( 30, 200,  30),   # rider+helm  – green
    4: ( 30,  30, 220),   # rider-noh   – red
}
GT_LABELS = {
    0: "GT: Helmet",
    1: "GT: Pedestrian",
    2: "GT: Motorcycle",
    3: "GT: Rider + Helmet",
    4: "GT: Rider - No Helmet",
}


# ── Geometry helpers ────────────────────────────────────────────────────────

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
    """Return all GT boxes as list of (cls, x1, y1, x2, y2)."""
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


def load_gt_riders(label_path: Path, img_w: int, img_h: int):
    """Return only rider boxes as {box, has_helmet}."""
    return [
        {"box": [x1, y1, x2, y2], "has_helmet": cls == 3}
        for cls, x1, y1, x2, y2 in load_gt_boxes(label_path, img_w, img_h)
        if cls in (3, 4)
    ]


# ── Drawing ─────────────────────────────────────────────────────────────────

def _put_label(canvas, text, x1, y1, color):
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    lx = max(0, min(int(x1), canvas.shape[1] - tw - 8))
    ly = max(th + 4, int(y1) - 4)
    cv2.rectangle(canvas, (lx, ly - th - 4), (lx + tw + 6, ly + 2), color, -1)
    cv2.putText(canvas, text, (lx + 3, ly - 2), font, scale, (255, 255, 255), thick)


def draw_gt(image: np.ndarray, gt_boxes) -> np.ndarray:
    """Draw ground-truth compliance boxes on a copy of the image."""
    canvas = image.copy()
    # Draw non-rider classes first (thinner), then riders on top (thicker)
    for priority in (False, True):
        for cls, x1, y1, x2, y2 in gt_boxes:
            is_rider = cls in (3, 4)
            if is_rider != priority:
                continue
            color = GT_COLORS.get(cls, (200, 200, 200))
            thick = 3 if is_rider else 2
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
            _put_label(canvas, GT_LABELS.get(cls, str(cls)), x1, y1, color)
    return canvas


def make_panel(gt_img: np.ndarray, pred_img: np.ndarray,
               title: str, subtitle: str = "") -> np.ndarray:
    """Stack GT and prediction side by side with a title bar."""
    h = max(gt_img.shape[0], pred_img.shape[0])
    w = gt_img.shape[1]

    def pad(img):
        ph = h - img.shape[0]
        return np.vstack([img, np.ones((ph, img.shape[1], 3), dtype=np.uint8) * 240]) if ph else img

    gt_img   = pad(gt_img)
    pred_img = pad(pad_img := cv2.resize(pred_img, (w, h)) if pred_img.shape[1] != w else pred_img)

    header_h = 44
    canvas = np.ones((h + header_h, w * 2, 3), dtype=np.uint8) * 245
    canvas[header_h:, :w]  = gt_img
    canvas[header_h:, w:]  = pred_img
    cv2.line(canvas, (w, 0), (w, h + header_h), (160, 160, 160), 2)

    # Header text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Ground Truth",       (12, 28), font, 0.72, (40, 40, 40), 2)
    cv2.putText(canvas, "System Output",      (w + 12, 28), font, 0.72, (40, 40, 40), 2)
    if subtitle:
        cv2.putText(canvas, subtitle, (w * 2 - 8 - cv2.getTextSize(subtitle, font, 0.5, 1)[0][0], 28),
                    font, 0.50, (100, 100, 100), 1)
    return canvas


# ── Rider matching ──────────────────────────────────────────────────────────

def match_riders(gt, pred):
    pairs = sorted(
        [(  _iou(gr["box"], pr["human"]["box"]), gi, pi)
         for gi, gr in enumerate(gt)
         for pi, pr in enumerate(pred)],
        reverse=True,
    )
    m_gt, m_pred, matches = set(), set(), []
    for iou, gi, pi in pairs:
        if iou < MATCH_IOU or gi in m_gt or pi in m_pred:
            continue
        matches.append((gi, pi))
        m_gt.add(gi); m_pred.add(pi)
    return matches, m_gt, m_pred


# ── Per-image statistics ────────────────────────────────────────────────────

def image_stats(gt_riders, pred_riders):
    matches, m_gt, m_pred = match_riders(gt_riders, pred_riders)
    tp = len(matches)
    fn = len(gt_riders) - len(m_gt)    # missed riders
    fp = len(pred_riders) - len(m_pred) # false riders

    h_tp = h_fp = h_fn = h_tn = 0
    for gi, pi in matches:
        gt_h  = gt_riders[gi]["has_helmet"]
        pr_h  = pred_riders[pi]["wearing_helmet"]
        if     gt_h and     pr_h: h_tp += 1
        elif   gt_h and not pr_h: h_fn += 1
        elif not gt_h and   pr_h: h_fp += 1
        else:                      h_tn += 1

    return {
        "rider_tp": tp, "rider_fp": fp, "rider_fn": fn,
        "helmet_tp": h_tp, "helmet_fp": h_fp,
        "helmet_fn": h_fn, "helmet_tn": h_tn,
        "n_gt": len(gt_riders), "n_pred": len(pred_riders),
    }


# ── Category selection logic ────────────────────────────────────────────────

def categorise(s):
    cats = []
    # 3.7 – generalisation
    if s["n_gt"] == 1 and s["rider_fn"] == 0 and s["rider_fp"] == 0 and s["helmet_fp"] == 0 and s["helmet_fn"] == 0:
        cats.append("good_single")
    if s["n_gt"] >= 2 and s["rider_fn"] == 0 and s["rider_fp"] == 0:
        cats.append("good_multi")
    if s["n_gt"] >= 2 and s["rider_fn"] >= 1:
        cats.append("challenging_missed")
    if s["n_gt"] >= 3:
        cats.append("challenging_crowd")
    # 3.8 – errors
    if s["rider_fn"] >= 1:
        cats.append("missed_rider")
    if s["helmet_fp"] >= 1:
        cats.append("false_safe")
    if s["helmet_fn"] >= 1:
        cats.append("false_no_helmet")
    if s["rider_fp"] >= 1:
        cats.append("false_rider")
    return cats


CATEGORY_DIRS = {
    # section 3.7
    "good_single":          OUTPUT_DIR / "section_3_7_generalisation" / "1_good_single_rider",
    "good_multi":           OUTPUT_DIR / "section_3_7_generalisation" / "2_good_multiple_riders",
    "challenging_missed":   OUTPUT_DIR / "section_3_7_generalisation" / "3_challenging_missed_riders",
    "challenging_crowd":    OUTPUT_DIR / "section_3_7_generalisation" / "4_challenging_crowd",
    # section 3.8
    "missed_rider":         OUTPUT_DIR / "section_3_8_errors" / "1_missed_rider",
    "false_safe":           OUTPUT_DIR / "section_3_8_errors" / "2_false_safe",
    "false_no_helmet":      OUTPUT_DIR / "section_3_8_errors" / "3_false_no_helmet",
    "false_rider":          OUTPUT_DIR / "section_3_8_errors" / "4_false_rider",
}

CATEGORY_SUBTITLES = {
    "good_single":        "Correct: single rider",
    "good_multi":         "Correct: multiple riders",
    "challenging_missed": "Challenge: missed riders (low recall)",
    "challenging_crowd":  "Challenge: crowded scene",
    "missed_rider":       "Error: missed rider (FN)",
    "false_safe":         "Error: false safe — undetected no-helmet (FP)",
    "false_no_helmet":    "Error: false no-helmet — helmeted rider flagged (FN)",
    "false_rider":        "Error: false rider — pedestrian misclassified (FP)",
}

# Scoring function: pick the most representative examples
CATEGORY_SCORE = {
    "good_single":        lambda s: s["helmet_tp"] + s["helmet_tn"],
    "good_multi":         lambda s: s["n_gt"],
    "challenging_missed": lambda s: s["rider_fn"],
    "challenging_crowd":  lambda s: s["n_gt"],
    "missed_rider":       lambda s: s["rider_fn"],
    "false_safe":         lambda s: s["helmet_fp"],
    "false_no_helmet":    lambda s: s["helmet_fn"],
    "false_rider":        lambda s: s["rider_fp"],
}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    from ultralytics import YOLO

    for d in CATEGORY_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_PATH.name}")
    model = YOLO(str(MODEL_PATH))

    image_files = sorted(
        f for f in TEST_IMAGES_DIR.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"Running inference on {len(image_files)} images …\n")

    # Collect (stats, img_path, gt_boxes, analysis) for all images
    records = []
    for img_path in image_files:
        label_path = GT_LABELS_DIR / (img_path.stem + ".txt")

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
        gt_riders = [
            {"box": [x1, y1, x2, y2], "has_helmet": cls == 3}
            for cls, x1, y1, x2, y2 in gt_boxes if cls in (3, 4)
        ]

        stats = image_stats(gt_riders, analysis["riders"])
        records.append({
            "img_path":  img_path,
            "gt_boxes":  gt_boxes,
            "analysis":  analysis,
            "stats":     stats,
            "img_w":     img_w,
            "img_h":     img_h,
        })

    print("Categorising and selecting examples …\n")

    # For each category, collect candidates and sort by score, take top N
    category_candidates: dict[str, list] = {k: [] for k in CATEGORY_DIRS}

    for rec in records:
        for cat in categorise(rec["stats"]):
            if cat in category_candidates:
                category_candidates[cat].append(rec)

    saved_counts = {}
    for cat, candidates in category_candidates.items():
        scorer = CATEGORY_SCORE[cat]
        limit  = EXAMPLES_PER_CATEGORY.get(cat, 3)
        ranked = sorted(candidates, key=lambda r: scorer(r["stats"]), reverse=True)
        top    = ranked if limit is None else ranked[:limit]
        saved_counts[cat] = len(top)

        for rank, rec in enumerate(top, 1):
            img_bgr  = cv2.imread(str(rec["img_path"]))
            if img_bgr is None:
                continue

            gt_img   = draw_gt(img_bgr, rec["gt_boxes"])
            pred_img = draw_analysis(img_bgr, rec["analysis"], show_labels=True)

            s   = rec["stats"]
            sub = (
                f"{CATEGORY_SUBTITLES[cat]}  |  "
                f"GT riders={s['n_gt']}  pred={s['n_pred']}  "
                f"FP={s['rider_fp']} FN={s['rider_fn']}  "
                f"false-safe={s['helmet_fp']} false-noh={s['helmet_fn']}"
            )
            panel = make_panel(gt_img, pred_img, cat, subtitle="")

            # Burn the subtitle below the header
            cv2.putText(panel, sub, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)

            out_path = CATEGORY_DIRS[cat] / f"{rank:02d}_{rec['img_path'].name}"
            cv2.imwrite(str(out_path), panel)

    # Summary
    print("=" * 65)
    print(f"{'Category':<35} {'Found':>6} {'Saved':>6}")
    print("=" * 65)
    for cat, count in saved_counts.items():
        total = len(category_candidates[cat])
        flag  = "  <-- none found!" if total == 0 else ""
        print(f"  {cat:<33} {total:>6} {count:>6}{flag}")

    print(f"\nAll images saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
