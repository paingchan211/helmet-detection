"""
review_annotations.py
---------------------
Fast review tool for checking and correcting pre-annotations.

Controls
--------
  ENTER / SPACE   Accept current image and move to next
  BACKSPACE       Go back to previous image
  Left-click      Cycle the clicked human/rider box through:
                    pedestrian (orange) → rider+helmet (green) → rider+no-helmet (red) → ...
  S               Save current image and move to next (same as ENTER)
  Q / ESC         Quit and save progress

Class colours
-------------
  Purple   helmet
  Blue     motorcycle
  Orange   human (pedestrian)
  Green    rider with helmet
  Red      rider without helmet
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT        = Path(__file__).parent
SRC_IMAGES  = ROOT / "test" / "images"
SRC_LABELS  = ROOT / "rules_evaluation" / "labels_annotated"   # output of pre_annotate.py
DST_LABELS  = ROOT / "rules_evaluation" / "labels_annotated"   # overwrite in place

CLS_HELMET     = 0
CLS_HUMAN      = 1
CLS_MOTORCYCLE = 2
CLS_RIDER_HELM = 3
CLS_RIDER_NOH  = 4

CLS_NAMES = {
    CLS_HELMET:     "helmet",
    CLS_HUMAN:      "pedestrian",
    CLS_MOTORCYCLE: "motorcycle",
    CLS_RIDER_HELM: "rider+helmet",
    CLS_RIDER_NOH:  "rider NO helmet",
}

# BGR colours
COLORS = {
    CLS_HELMET:     (200,  40, 180),  # purple
    CLS_HUMAN:      ( 40, 160, 235),  # orange
    CLS_MOTORCYCLE: (240, 130,  60),  # blue
    CLS_RIDER_HELM: ( 50, 200,  50),  # green
    CLS_RIDER_NOH:  ( 40,  40, 220),  # red
}

# Cycle order when clicking a human/rider box
CYCLE = [CLS_HUMAN, CLS_RIDER_HELM, CLS_RIDER_NOH]

WINDOW = "Annotation Review  |  ENTER=accept  CLICK=change class  BACKSPACE=prev  Q=quit"
MAX_DISPLAY_H = 800


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_labels(path: Path, img_w: int, img_h: int):
    """Return list of [cls, x1, y1, x2, y2] in pixel coords."""
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        boxes.append([cls, x1, y1, x2, y2])
    return boxes


def save_labels(path: Path, boxes: list, img_w: int, img_h: int):
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + "\n")


def draw(image: np.ndarray, boxes: list, scale: float) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]

    for cls, x1, y1, x2, y2 in boxes:
        sx1, sy1 = int(x1 * scale), int(y1 * scale)
        sx2, sy2 = int(x2 * scale), int(y2 * scale)
        color = COLORS.get(cls, (200, 200, 200))
        if cls in (CLS_RIDER_HELM, CLS_RIDER_NOH):
            thick = 3
        else:
            thick = 2

        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), color, thick)
        label = CLS_NAMES.get(cls, str(cls))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        lx = max(0, min(sx1, w - tw - 6))
        ly = max(th + 4, sy1 - 4)
        cv2.rectangle(canvas, (lx, ly - th - 4), (lx + tw + 6, ly + 2), color, -1)
        cv2.putText(canvas, label, (lx + 3, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return canvas


def hit_box(mx: int, my: int, boxes: list, scale: float) -> int | None:
    """Return index of topmost human/rider box that contains (mx, my)."""
    for i in range(len(boxes) - 1, -1, -1):
        cls, x1, y1, x2, y2 = boxes[i]
        if cls not in (CLS_HUMAN, CLS_RIDER_HELM, CLS_RIDER_NOH):
            continue
        sx1, sy1 = int(x1 * scale), int(y1 * scale)
        sx2, sy2 = int(x2 * scale), int(y2 * scale)
        if sx1 <= mx <= sx2 and sy1 <= my <= sy2:
            return i
    return None


# ---------------------------------------------------------------------------
# Mouse callback state
# ---------------------------------------------------------------------------

_click_position: tuple[int, int] | None = None


def _on_mouse(event, x, y, flags, param):
    global _click_position
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_position = (x, y)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    global _click_position

    if not SRC_LABELS.exists():
        print("labels_annotated/ not found. Run  python pre_annotate.py  first.")
        sys.exit(1)

    image_files = sorted(
        f for f in SRC_IMAGES.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_files:
        print(f"No images found in {SRC_IMAGES}")
        sys.exit(1)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, _on_mouse)

    idx = 0
    # Per-image boxes cache so edits survive navigation
    cache: dict[int, list] = {}

    while 0 <= idx < len(image_files):
        img_path = image_files[idx]
        label_path = SRC_LABELS / (img_path.stem + ".txt")

        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            idx += 1
            continue
        oh, ow = img_orig.shape[:2]

        # Scale for display
        scale = min(1.0, MAX_DISPLAY_H / oh)
        dw, dh = int(ow * scale), int(oh * scale)
        img_display = cv2.resize(img_orig, (dw, dh))

        if idx not in cache:
            if label_path.exists():
                cache[idx] = load_labels(label_path, ow, oh)
            else:
                cache[idx] = []

        boxes = cache[idx]
        _click_position = None
        needs_redraw = True

        while True:
            if needs_redraw:
                frame = draw(img_display, boxes, scale)
                # Status bar
                total = len(image_files)
                n_rh = 0
                n_rn = 0
                n_p = 0
                for box in boxes:
                    class_id = box[0]
                    if class_id == CLS_RIDER_HELM:
                        n_rh += 1
                    elif class_id == CLS_RIDER_NOH:
                        n_rn += 1
                    elif class_id == CLS_HUMAN:
                        n_p += 1

                status = (f"  {idx+1}/{total}  {img_path.name}"
                          f"   riders+helm={n_rh}  riders-noh={n_rn}  pedestrian={n_p}"
                          f"   ENTER=accept  CLICK=cycle class  BACK=prev  Q=quit")
                bar = np.zeros((32, frame.shape[1], 3), dtype=np.uint8)
                cv2.putText(bar, status, (6, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
                combined = np.vstack([bar, frame])
                cv2.imshow(WINDOW, combined)
                cv2.resizeWindow(WINDOW, dw, dh + 32)
                needs_redraw = False

            key = cv2.waitKey(30) & 0xFF

            # Handle click
            if _click_position is not None:
                mx, my = _click_position
                my -= 32  # subtract status bar height
                _click_position = None
                hit = hit_box(mx, my, boxes, scale)
                if hit is not None:
                    cur_cls = boxes[hit][0]
                    next_cls = CYCLE[(CYCLE.index(cur_cls) + 1) % len(CYCLE)]
                    boxes[hit][0] = next_cls
                    needs_redraw = True
                continue

            if key in (13, 32, ord('s'), ord('S')):   # ENTER / SPACE / S → accept
                if label_path.exists() or boxes:
                    save_labels(label_path, boxes, ow, oh)
                idx += 1
                break

            elif key == 8:                              # BACKSPACE → previous
                if label_path.exists() or boxes:
                    save_labels(label_path, boxes, ow, oh)
                idx = max(0, idx - 1)
                break

            elif key in (27, ord('q'), ord('Q')):      # ESC / Q → quit
                if label_path.exists() or boxes:
                    save_labels(label_path, boxes, ow, oh)
                print(f"\nSaved progress at image {idx + 1}/{len(image_files)}.")
                print("Re-run to continue from where you left off.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print(f"\nAll {len(image_files)} images reviewed. Labels saved to {DST_LABELS}")


if __name__ == "__main__":
    main()
