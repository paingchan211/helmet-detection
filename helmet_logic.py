"""
helmet_logic.py
---------------
Analyses raw object-detection results to determine which humans are riding
motorcycles and whether each rider is wearing a helmet.

Design notes
~~~~~~~~~~~~
* All thresholds and weights are named constants – tune them in one place.
* Motorcycles can be shared by multiple matched humans so passengers on the
  same motorcycle are counted as riders.
* Helmets are assigned exclusively (one-to-one) so that a single helmet can
  never be claimed by two riders simultaneously.
* Detections are normalised (class name lowercased, box validated) at ingestion
  so the rest of the pipeline can assume clean data.
* A lightweight per-class Non-Maximum Suppression (NMS) step removes duplicate
  boxes produced by some detectors before any association logic runs.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------

HELMET_CLASS = "helmet"
HUMAN_CLASS = "human"
MOTORCYCLE_CLASS = "motorcycle"

# ---------------------------------------------------------------------------
# Tunable thresholds & weights
# ---------------------------------------------------------------------------

# _rider_score must reach this value for a human to be considered a rider.
# Lower → more false positives; higher → missed riders near frame edges.
RIDER_SCORE_THRESHOLD: float = 0.54

# Helmet association score must reach this value to count as "wearing helmet".
# Based on intersection-over-helmet-area + positional bonus.
HELMET_SCORE_THRESHOLD: float = 0.28

# Contribution of IoU(expanded_human, motorcycle) to the rider score.
# Keep higher than DISTANCE_WEIGHT so spatial overlap dominates.
RIDER_OVERLAP_WEIGHT: float = 2.0

# Multiplier applied to the x-axis component of the centre-distance score.
# Horizontal proximity matters more than vertical (rider sits above the bike).
RIDER_X_DISTANCE_WEIGHT: float = 0.75

# Multiplier applied to the y-axis component of the centre-distance score.
RIDER_Y_DISTANCE_WEIGHT: float = 0.45

# Positional bonus added to the helmet score when the helmet centre sits
# inside the computed head region.
HELMET_POSITION_BONUS: float = 0.35

# IoU threshold for Non-Maximum Suppression.  Boxes with IoU > this value
# are considered duplicates and the lower-confidence one is removed.
NMS_IOU_THRESHOLD: float = 0.45

# ---------------------------------------------------------------------------
# Visualisation colours  (BGR for OpenCV)
# ---------------------------------------------------------------------------

COLOR_HELMET = (180, 40, 200)
COLOR_HUMAN = (235, 160, 40)
COLOR_MOTORCYCLE = (60, 130, 240)
COLOR_RIDER_SAFE = (30, 170, 70)
COLOR_RIDER_DANGER = (30, 30, 230)
COLOR_LABEL_TEXT = (255, 255, 255)
COLOR_LABEL_SHADOW = (20, 24, 31)


# ===========================================================================
# Internal geometry helpers
# ===========================================================================

def _area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _intersection(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a: list[float], b: list[float]) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union else 0.0


def _expand(
    box: list[float],
    x_factor: float,
    y_top: float,
    y_bottom: float,
) -> list[float]:
    """Return *box* expanded by fractional amounts on each side."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return [
        x1 - w * x_factor,
        y1 - h * y_top,
        x2 + w * x_factor,
        y2 + h * y_bottom,
    ]


def _head_region(human_box: list[float]) -> list[float]:
    """
    Return a box that covers the likely head area of a human detection.

    Extends slightly beyond the top and sides of the human box to handle
    helmets whose bounding box protrudes above the person silhouette.
    """
    x1, y1, x2, y2 = human_box
    w, h = x2 - x1, y2 - y1
    return [
        x1 - 0.2 * w,
        y1 - 0.1 * h,
        x2 + 0.2 * w,
        y1 + 0.45 * h,
    ]


def _helmet_score(head_box: list[float], helmet_box: list[float]) -> float:
    hc = _center(helmet_box)
    in_head = (
        head_box[0] <= hc[0] <= head_box[2]
        and head_box[1] <= hc[1] <= head_box[3]
    )
    overlap = _intersection(head_box, helmet_box) / max(1.0, _area(helmet_box))
    return overlap + (HELMET_POSITION_BONUS if in_head else 0.0)


def _draw_label(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    thickness: int,
) -> None:
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    text_w, text_h = text_size
    pad_x, pad_y = 5, 4
    x, baseline_y = origin
    x = max(0, min(x, canvas.shape[1] - text_w - pad_x * 2))
    baseline_y = max(text_h + pad_y, min(baseline_y, canvas.shape[0] - pad_y))
    top_left = (x, baseline_y - text_h - pad_y * 2)
    bottom_right = (x + text_w + pad_x * 2, baseline_y + baseline)
    cv2.rectangle(canvas, top_left, bottom_right, color, -1)
    cv2.rectangle(canvas, top_left, bottom_right, COLOR_LABEL_SHADOW, 1)
    cv2.putText(
        canvas,
        text,
        (x + pad_x, baseline_y - pad_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        COLOR_LABEL_TEXT,
        thickness,
    )


# ===========================================================================
# Non-Maximum Suppression
# ===========================================================================

def _nms(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove duplicate detections within a single class using IoU-based NMS.

    Detections are assumed to already share the same class.  The one with the
    higher confidence is kept when two boxes overlap beyond NMS_IOU_THRESHOLD.
    """
    if len(detections) <= 1:
        return detections

    # Sort descending by confidence so we always keep the best box.
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []

    for candidate in sorted_dets:
        suppressed = any(
            _iou(candidate["box"], kept_det["box"]) > NMS_IOU_THRESHOLD
            for kept_det in kept
        )
        if not suppressed:
            kept.append(candidate)

    return kept


# ===========================================================================
# Ingestion & normalisation
# ===========================================================================

def _normalise(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return a cleaned copy of the detection list.

    * Class names are lower-cased and stripped.
    * Boxes are reordered so that (x1 < x2, y1 < y2).
    * Detections with zero-area boxes are dropped.
    """
    normalised: list[dict[str, Any]] = []
    for det in detections:
        class_name = str(det.get("class_name", "")).strip().lower()
        box = list(det.get("box", []))
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        # Ensure top-left / bottom-right ordering.
        box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        if _area(box) <= 0:
            continue
        normalised.append({**det, "class_name": class_name, "box": box})
    return normalised


# ===========================================================================
# Rider–motorcycle association score
# ===========================================================================

def _rider_score(human_box: list[float], motorcycle_box: list[float]) -> float:
    """
    Score how likely *human_box* belongs to the rider of *motorcycle_box*.

    Score = overlap_component + distance_component

    overlap_component:
        IoU between an *expanded* human box and the motorcycle box, weighted
        by RIDER_OVERLAP_WEIGHT.  Expanding the human box downward captures
        cases where the rider sits atop the bike with little direct overlap.

    distance_component:
        1 − normalised centre distance, clamped to [0, 1].  Horizontal and
        vertical distances are weighted separately because a rider's centre
        is typically above the motorcycle's centre.
    """
    hx, hy = _center(human_box)
    mx, my = _center(motorcycle_box)
    human_w = max(1.0, human_box[2] - human_box[0])
    human_h = max(1.0, human_box[3] - human_box[1])
    bike_w = max(1.0, motorcycle_box[2] - motorcycle_box[0])
    bike_h = max(1.0, motorcycle_box[3] - motorcycle_box[1])

    expanded_human = _expand(human_box, x_factor=0.45, y_top=0.15, y_bottom=0.75)
    overlap = _iou(expanded_human, motorcycle_box)

    x_close = abs(hx - mx) / max(human_w, bike_w)
    y_close = abs(hy - my) / max(human_h, bike_h)
    distance_score = max(
        0.0,
        1.0 - (x_close * RIDER_X_DISTANCE_WEIGHT + y_close * RIDER_Y_DISTANCE_WEIGHT),
    )

    return overlap * RIDER_OVERLAP_WEIGHT + distance_score


# ===========================================================================
# Main analysis
# ===========================================================================

def analyze_detections(
    raw_detections: list[dict[str, Any]],
    rider_score_threshold: float = RIDER_SCORE_THRESHOLD,
    helmet_score_threshold: float = HELMET_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """
    Pair humans with motorcycles, then pair riders with helmets.

    Motorcycles can be assigned to multiple humans because one motorcycle may
    carry a driver and passengers.  Helmets are still assigned to **at most
    one** rider, preventing the same helmet from being counted twice.

    Parameters
    ----------
    raw_detections:
        List of detection dicts, each containing at minimum:
        ``{"class_name": str, "box": [x1, y1, x2, y2], "confidence": float}``

    Returns
    -------
    dict with keys:
        ``detections`` – cleaned detections used for analysis
        ``riders``     – list of rider dicts (see below)
        ``counts``     – summary counters
    """
    # --- 1. Normalise and deduplicate ---
    detections = _normalise(raw_detections)

    # Group by class for NMS, then flatten back.
    class_groups: dict[str, list[dict[str, Any]]] = {}
    for det in detections:
        class_groups.setdefault(det["class_name"], []).append(det)

    deduped: list[dict[str, Any]] = []
    for class_name, group in class_groups.items():
        deduped.extend(_nms(group))

    helmets = [d for d in deduped if d["class_name"] == HELMET_CLASS]
    humans = [d for d in deduped if d["class_name"] == HUMAN_CLASS]
    motorcycles = [d for d in deduped if d["class_name"] == MOTORCYCLE_CLASS]

    # --- 2. Associate humans with motorcycles ---
    # A motorcycle can carry multiple people, so do not remove it after a
    # successful match.  Each human simply chooses the best motorcycle nearby.
    riders: list[dict[str, Any]] = []
    human_scores: list[dict[str, Any]] = []

    for index, human in enumerate(humans, start=1):
        human_box = human["box"]
        head_box = _head_region(human_box)

        best_motorcycle = None
        best_score = 0.0

        for motorcycle in motorcycles:
            score = _rider_score(human_box, motorcycle["box"])
            if score > best_score:
                best_score = score
                best_motorcycle = motorcycle

        best_helmet_score = max(
            (_helmet_score(head_box, helmet["box"]) for helmet in helmets),
            default=0.0,
        )
        human_scores.append(
            {
                "id": index,
                "human": human,
                "rider_score": round(best_score, 3),
                "helmet_score": round(best_helmet_score, 3),
            }
        )

        if best_motorcycle is None or best_score < rider_score_threshold:
            continue

        # --- 3. Associate rider with a helmet (exclusive) ---
        riders.append(
            {
                "id": index,
                "human": human,
                "motorcycle": best_motorcycle,
                "head_box": head_box,
                "helmet": None,
                "wearing_helmet": False,
                "rider_score": round(best_score, 3),
                "helmet_score": 0.0,
            }
        )

    # Assign helmets to riders exclusively (highest-scoring pair first).
    # Build all (rider_idx, helmet_idx, score) triples, sort by score desc,
    # then assign greedily.
    helmet_scores: list[tuple[float, int, int]] = []
    available_helmets = list(range(len(helmets)))

    for rider_idx, rider in enumerate(riders):
        head_box = rider["head_box"]
        for helmet_idx, helmet in enumerate(helmets):
            score = _helmet_score(head_box, helmet["box"])
            helmet_scores.append((score, rider_idx, helmet_idx))

    helmet_scores.sort(reverse=True)  # best matches first

    assigned_helmet_indices: set[int] = set()
    assigned_rider_indices: set[int] = set()

    for score, rider_idx, helmet_idx in helmet_scores:
        if rider_idx in assigned_rider_indices:
            continue
        if helmet_idx in assigned_helmet_indices:
            continue
        if score < helmet_score_threshold:
            break  # remaining scores are only worse

        riders[rider_idx]["helmet"] = helmets[helmet_idx]
        riders[rider_idx]["wearing_helmet"] = True
        riders[rider_idx]["helmet_score"] = round(score, 3)
        assigned_rider_indices.add(rider_idx)
        assigned_helmet_indices.add(helmet_idx)

    return {
        "detections": deduped,
        "riders": riders,
        "human_scores": human_scores,
        "counts": {
            "helmets": len(helmets),
            "humans": len(humans),
            "motorcycles": len(motorcycles),
            "riders": len(riders),
            "with_helmet": sum(1 for r in riders if r["wearing_helmet"]),
            "without_helmet": sum(1 for r in riders if not r["wearing_helmet"]),
        },
        "thresholds": {
            "rider_score": rider_score_threshold,
            "helmet_score": helmet_score_threshold,
        },
    }


# ===========================================================================
# Summary
# ===========================================================================

def summarize_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    """
    Return a flat summary dict suitable for logging or display.

    compliance_percent is 0.0 when there are no riders (avoids division by zero
    without relying on a dummy ``max(1, ...)`` guard that masks the intent).
    """
    counts = analysis["counts"]
    detections = analysis["detections"]
    confidences = [float(d["confidence"]) for d in detections]

    compliance = (
        round((counts["with_helmet"] / counts["riders"]) * 100, 1)
        if counts["riders"]
        else 0.0
    )

    return {
        **counts,
        "detection_count": len(detections),
        "average_confidence": (
            round(sum(confidences) / len(confidences) * 100, 1) if confidences else 0.0
        ),
        "lowest_confidence": (
            round(min(confidences) * 100, 1) if confidences else 0.0
        ),
        "compliance_percent": compliance,
        "risk_level": "High" if counts["without_helmet"] else "Low",
    }


# ===========================================================================
# Visualisation
# ===========================================================================

def draw_analysis(
    image: np.ndarray,
    analysis: dict[str, Any],
    show_labels: bool = False,
) -> np.ndarray:
    """
    Draw bounding boxes onto *image*.

    * Green boxes  → helmets
    * Blue boxes   → humans (raw detections)
    * Amber boxes  → motorcycles
    * Thick green outline → rider wearing helmet
    * Thick red outline   → rider without helmet
    * Labels are optional so images can use browser hover labels while videos
      can keep burned-in labels.
    """
    canvas = image.copy()

    color_map = {
        HELMET_CLASS: COLOR_HELMET,
        HUMAN_CLASS: COLOR_HUMAN,
        MOTORCYCLE_CLASS: COLOR_MOTORCYCLE,
    }

    # Draw all raw detections first (thin boxes).
    for det in analysis["detections"]:
        color = color_map.get(det["class_name"], (200, 200, 200))
        x1, y1, x2, y2 = (int(v) for v in det["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if show_labels:
            label = f"{det['class_name']} {det['confidence']:.2f}"
            _draw_label(
                canvas,
                label,
                (x1, max(18, y1 - 8)),
                color,
                0.55,
                2,
            )

    # Overlay thick rider boxes with compliance status.
    for rider in analysis["riders"]:
        x1, y1, x2, y2 = (int(v) for v in rider["human"]["box"])
        wearing = rider["wearing_helmet"]
        color = COLOR_RIDER_SAFE if wearing else COLOR_RIDER_DANGER
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 4)
        if show_labels:
            status = "HELMET" if wearing else "WITHOUT HELMET"
            _draw_label(
                canvas,
                status,
                (x1, min(canvas.shape[0] - 12, y2 + 24)),
                color,
                0.75,
                2,
            )

    return canvas
