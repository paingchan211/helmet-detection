"""
eval_compliance_v2.py
---------------------
Evaluates the rule-based rider/helmet compliance logic for all 8 YOLO models
against MANUALLY REVIEWED ground-truth annotations in test/labels_annotated/.

Ground-truth classes
--------------------
  0  helmet
  1  human (pedestrian, not a rider)
  2  motorcycle
  3  rider_with_helmet
  4  rider_without_helmet

Evaluation logic
----------------
For each image:
  1. Load GT riders (class 3 / 4) from labels_annotated/
  2. Run YOLO inference -> analyze_detections() -> predicted riders
  3. Match predicted riders to GT riders by human-box IoU (greedy, best first)
  4. Score as TP/FP/FN for rider detection and helmet compliance

Metrics reported per model
--------------------------
Rider detection   : precision, recall, F1
Helmet compliance : accuracy, false-safe rate, false-no-helmet rate
  false safe      : GT=no helmet  predicted=helmet   (dangerous - unhelmetted rider marked safe)
  false no-helmet : GT=helmet     predicted=no helmet (unfair   - helmeted rider flagged)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from helmet_logic import analyze_detections

TEST_IMAGES_DIR  = ROOT / "test" / "images"
GT_LABELS_DIR    = ROOT / "rules_evaluation" / "labels_annotated"
MODELS_DIR       = ROOT / "exported_models"
OUTPUT_CSV       = ROOT / "test_runs" / "compliance_eval_v2.csv"

CLASS_NAMES = ["helmet", "human", "motorcycle"]
CONF_THRESHOLD = 0.35
MATCH_IOU      = 0.30   # min IoU to pair a predicted rider to a GT rider

MODELS = [
    "helmet_data_yolov8n_e60_best.pt",
    "helmet_data_yolov8n_e100_best.pt",
    "helmet_data_yolov8s_e60_best.pt",
    "helmet_data_yolov8s_e100_best.pt",
    "helmet_data_combined_yolov8n_e60_best.pt",
    "helmet_data_combined_yolov8n_e100_best.pt",
    "helmet_data_combined_yolov8s_e60_best.pt",
    "helmet_data_combined_yolov8s_e100_best.pt",
]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    if union == 0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def load_gt_riders(label_path: Path, img_w: int, img_h: int):
    """
    Return list of dicts  {box: [x1,y1,x2,y2], has_helmet: bool}
    from the manually reviewed labels_annotated/ file.
    """
    riders = []
    if not label_path.exists():
        return riders
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        if cls not in (3, 4):
            continue                        # skip helmets, motorcycles, pedestrians
        cx, cy, bw, bh = (float(p) for p in parts[1:])
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        riders.append({"box": [x1, y1, x2, y2], "has_helmet": cls == 3})
    return riders


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_riders(gt: list[dict], pred: list[dict]):
    """
    Greedy best-first matching by human-box IoU.
    Returns (matches [(gt_i, pred_i)], matched_gt_set, matched_pred_set).
    """
    pairs = []
    for gi, gr in enumerate(gt):
        for pi, pr in enumerate(pred):
            iou = _iou(gr["box"], pr["human"]["box"])
            pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)

    matched_gt:   set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int]] = []

    for iou, gi, pi in pairs:
        if iou < MATCH_IOU:
            break
        if gi in matched_gt or pi in matched_pred:
            continue
        matches.append((gi, pi))
        matched_gt.add(gi)
        matched_pred.add(pi)

    return matches, matched_gt, matched_pred


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model_name: str) -> dict:
    from ultralytics import YOLO

    print(f"  {model_name}", flush=True)
    model = YOLO(str(MODELS_DIR / model_name))

    rider_tp = rider_fp = rider_fn = 0
    helmet_tp = helmet_fp = helmet_fn = helmet_tn = 0
    total_gt = total_pred = 0

    image_files = sorted(
        f for f in TEST_IMAGES_DIR.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    for img_path in image_files:
        label_path = GT_LABELS_DIR / (img_path.stem + ".txt")

        # --- YOLO inference ---
        result = model(str(img_path), verbose=False, conf=CONF_THRESHOLD)[0]
        img_h, img_w = result.orig_shape

        pred_dets = []
        if result.boxes is not None:
            for box in result.boxes:
                pred_dets.append({
                    "class_name": CLASS_NAMES[int(box.cls[0])],
                    "box":        box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                })

        pred_riders = analyze_detections(pred_dets)["riders"]
        gt_riders   = load_gt_riders(label_path, img_w, img_h)

        total_gt   += len(gt_riders)
        total_pred += len(pred_riders)

        matches, matched_gt, matched_pred = match_riders(gt_riders, pred_riders)

        rider_tp += len(matches)
        rider_fn += len(gt_riders)   - len(matched_gt)
        rider_fp += len(pred_riders) - len(matched_pred)

        for gi, pi in matches:
            gt_h  = gt_riders[gi]["has_helmet"]
            pr_h  = pred_riders[pi]["wearing_helmet"]
            if     gt_h and     pr_h: helmet_tp += 1
            elif   gt_h and not pr_h: helmet_fn += 1   # false no-helmet
            elif not gt_h and   pr_h: helmet_fp += 1   # false safe
            else:                     helmet_tn += 1

    # --- Aggregate metrics ---
    prec = safe_divide(rider_tp, rider_tp + rider_fp)
    rec = safe_divide(rider_tp, rider_tp + rider_fn)
    f1 = safe_divide(2 * prec * rec, prec + rec)

    matched = rider_tp
    helm_acc = safe_divide(helmet_tp + helmet_tn, matched)
    false_safe = safe_divide(helmet_fp, helmet_fp + helmet_tn)
    false_noh = safe_divide(helmet_fn, helmet_fn + helmet_tp)
    helm_prec = safe_divide(helmet_tp, helmet_tp + helmet_fp)
    helm_rec = safe_divide(helmet_tp, helmet_tp + helmet_fn)
    helm_f1 = safe_divide(2 * helm_prec * helm_rec, helm_prec + helm_rec)

    short = model_name.replace("helmet_data_", "").replace("_best.pt", "")
    print(
        f"    Rider   P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}"
        f"  |  Helmet acc={helm_acc:.4f}"
        f"  false-safe={false_safe:.4f}"
        f"  false-noh={false_noh:.4f}"
    )

    return {
        "model":           short,
        "gt_riders":       total_gt,
        "pred_riders":     total_pred,
        "rider_tp":        rider_tp,
        "rider_fp":        rider_fp,
        "rider_fn":        rider_fn,
        "rider_precision": round(prec,  4),
        "rider_recall":    round(rec,   4),
        "rider_f1":        round(f1,    4),
        "matched_riders":  matched,
        "helmet_tp":       helmet_tp,
        "helmet_fp":       helmet_fp,
        "helmet_fn":       helmet_fn,
        "helmet_tn":       helmet_tn,
        "helmet_accuracy": round(helm_acc,   4),
        "helmet_precision":round(helm_prec,  4),
        "helmet_recall":   round(helm_rec,   4),
        "helmet_f1":       round(helm_f1,    4),
        "false_safe_rate": round(false_safe, 4),
        "false_noh_rate":  round(false_noh,  4),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"Ground-truth labels : {GT_LABELS_DIR}")
    print(f"Test images         : {TEST_IMAGES_DIR}")
    print(f"Models              : {len(MODELS)}")
    print(f"Output CSV          : {OUTPUT_CSV}\n")

    results = []
    for model_name in MODELS:
        result = evaluate_model(model_name)
        results.append(result)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {OUTPUT_CSV}")

    # Pretty summary table
    w = 36
    print("\n" + "=" * 100)
    print(f"{'Model':<{w}} {'RiderP':>7} {'RiderR':>7} {'RiderF1':>8}"
          f" {'HelmAcc':>8} {'HelmF1':>7} {'FalSafe':>8} {'FalNoH':>8}")
    print("=" * 100)
    def rider_f1(row: dict) -> float:
        return row["rider_f1"]

    sorted_results = sorted(results, key=rider_f1, reverse=True)
    for r in sorted_results:
        print(
            f"{r['model']:<{w}}"
            f" {r['rider_precision']:>7.4f} {r['rider_recall']:>7.4f} {r['rider_f1']:>8.4f}"
            f" {r['helmet_accuracy']:>8.4f} {r['helmet_f1']:>7.4f}"
            f" {r['false_safe_rate']:>8.4f} {r['false_noh_rate']:>8.4f}"
        )

    best = sorted_results[0]
    print(f"\nBest overall (rider F1): {best['model']}")


if __name__ == "__main__":
    main()
