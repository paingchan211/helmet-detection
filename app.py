from __future__ import annotations

import os
import queue
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename

from helmet_logic import (
    HELMET_SCORE_THRESHOLD,
    RIDER_SCORE_THRESHOLD,
    analyze_detections,
    draw_analysis,
    summarize_analysis,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "best.pt"))
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS
DEFAULT_CONFIDENCE = 0.35
DEFAULT_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "640"))
DEFAULT_VIDEO_SLOWDOWN_FACTOR = 2.0

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "helmet-detection-dev")

_model: YOLO | None = None
_job_queue: queue.Queue[str] = queue.Queue()
_jobs: dict[str, "ProcessingJob"] = {}
_jobs_lock = threading.Lock()
_worker_started = False
_active_job_id: str | None = None


@dataclass
class ProcessingJob:
    id: str
    input_path: Path
    kind: str
    confidence: float
    rider_score_threshold: float
    helmet_score_threshold: float
    status: str = "queued"
    result: dict[str, Any] | None = None
    error: str | None = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Put your Colab best.pt file there or set MODEL_PATH."
            )
        _model = YOLO(str(MODEL_PATH))
    return _model


def allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in ALLOWED_IMAGE_EXTENSIONS:
        return "image"
    if ext in ALLOWED_VIDEO_EXTENSIONS:
        return "video"
    raise ValueError("Unsupported file type. Choose an image or video file.")


def confidence_from_form() -> float:
    raw_value = request.form.get("confidence", str(DEFAULT_CONFIDENCE))
    try:
        confidence = float(raw_value)
    except ValueError:
        return DEFAULT_CONFIDENCE
    return min(0.95, max(0.05, confidence))


def threshold_from_form(field_name: str, default: float) -> float:
    raw_value = request.form.get(field_name, str(default))
    try:
        threshold = float(raw_value)
    except ValueError:
        return default
    return min(1.5, max(0.0, threshold))


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def existing_upload_path(filename: str) -> Path:
    safe_name = Path(filename).name
    path = (UPLOAD_DIR / safe_name).resolve()
    upload_root = UPLOAD_DIR.resolve()
    if upload_root not in path.parents or not path.exists():
        raise ValueError("Uploaded file is no longer available.")
    return path


def result_to_detections(result: Any) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if result.boxes is None:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    names = result.names

    for box, class_id, confidence in zip(boxes, classes, confidences):
        detections.append(
            {
                "box": [float(v) for v in box.tolist()],
                "class_id": int(class_id),
                "class_name": names[int(class_id)],
                "confidence": float(confidence),
            }
        )
    return detections


def run_frame(
    frame,
    confidence: float,
    rider_score_threshold: float,
    helmet_score_threshold: float,
    show_labels: bool = False,
):
    model = get_model()
    result = model.predict(frame, conf=confidence, imgsz=DEFAULT_IMAGE_SIZE, verbose=False)[0]
    detections = result_to_detections(result)
    analysis = analyze_detections(
        detections,
        rider_score_threshold=rider_score_threshold,
        helmet_score_threshold=helmet_score_threshold,
    )
    annotated = draw_analysis(frame, analysis, show_labels=show_labels)
    return annotated, analysis


def output_url(output_name: str, output_path: Path) -> str:
    version = output_path.stat().st_mtime_ns
    return f"/static/outputs/{output_name}?v={version}"


def clear_previous_outputs(path: Path) -> None:
    for existing in OUTPUT_DIR.glob(f"{path.stem}_detected*"):
        existing.unlink(missing_ok=True)
    for existing in OUTPUT_DIR.glob(f"{path.stem}_conf_*_detected*"):
        existing.unlink(missing_ok=True)


def _box_key(box: list[float]) -> tuple[int, int, int, int]:
    return tuple(round(v) for v in box)


def _box_kind(class_name: str) -> str:
    kind = "".join(ch if ch.isalnum() else "-" for ch in class_name.lower()).strip("-")
    return kind or "object"


def _display_class_name(class_name: str) -> str:
    return class_name.replace("_", " ").replace("-", " ").title()


def _percent_box(box: list[float], width: int, height: int) -> dict[str, float]:
    x1, y1, x2, y2 = box
    x1 = min(width, max(0.0, x1))
    y1 = min(height, max(0.0, y1))
    x2 = min(width, max(0.0, x2))
    y2 = min(height, max(0.0, y2))
    return {
        "left": x1 / width * 100,
        "top": y1 / height * 100,
        "width": (x2 - x1) / width * 100,
        "height": (y2 - y1) / height * 100,
    }


def _score_metric(label: str, score: float, threshold: float, scale: float) -> dict[str, Any]:
    return {
        "label": label,
        "score": round(score, 3),
        "threshold": threshold,
        "score_percent": min(100.0, score / scale * 100),
        "threshold_percent": min(100.0, threshold / scale * 100),
        "passed": score >= threshold,
    }


def _threshold_scales(
    human_scores: list[dict[str, Any]],
    rider_score_threshold: float,
    helmet_score_threshold: float,
) -> tuple[float, float]:
    rider_scores = [float(human["rider_score"]) for human in human_scores]
    helmet_scores = [float(human["helmet_score"]) for human in human_scores]
    rider_scale = max(1.0, rider_score_threshold * 1.15, *(score * 1.15 for score in rider_scores))
    helmet_scale = max(1.0, helmet_score_threshold * 1.15, *(score * 1.15 for score in helmet_scores))
    return rider_scale, helmet_scale


def _human_threshold_metrics(
    human_score: dict[str, Any],
    rider_scale: float,
    helmet_scale: float,
    rider_score_threshold: float,
    helmet_score_threshold: float,
) -> list[dict[str, Any]]:
    return [
        _score_metric("Rider", float(human_score["rider_score"]), rider_score_threshold, rider_scale),
        _score_metric("Helmet", float(human_score["helmet_score"]), helmet_score_threshold, helmet_scale),
    ]


def hover_boxes_for_analysis(
    analysis: dict[str, Any],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    thresholds = analysis.get("thresholds", {})
    rider_score_threshold = float(thresholds.get("rider_score", RIDER_SCORE_THRESHOLD))
    helmet_score_threshold = float(thresholds.get("helmet_score", HELMET_SCORE_THRESHOLD))
    rider_boxes = {_box_key(rider["human"]["box"]) for rider in analysis["riders"]}
    human_score_by_box = {
        _box_key(human_score["human"]["box"]): human_score
        for human_score in analysis["human_scores"]
    }
    rider_scale, helmet_scale = _threshold_scales(
        analysis["human_scores"],
        rider_score_threshold,
        helmet_score_threshold,
    )
    hover_boxes: list[dict[str, Any]] = []

    for det in analysis["detections"]:
        display_class = _display_class_name(det["class_name"])
        box_key = _box_key(det["box"])
        if det["class_name"] == "human" and box_key in rider_boxes:
            continue
        human_score = human_score_by_box.get(box_key)
        hover_boxes.append(
            {
                **_percent_box(det["box"], width, height),
                "label": f"{display_class} {det['confidence']:.2f}",
                "kind": _box_kind(det["class_name"]),
                "metrics": (
                    _human_threshold_metrics(
                        human_score,
                        rider_scale,
                        helmet_scale,
                        rider_score_threshold,
                        helmet_score_threshold,
                    )
                    if human_score
                    else []
                ),
            }
        )

    for rider in analysis["riders"]:
        human = rider["human"]
        status = "Helmet" if rider["wearing_helmet"] else "Without Helmet"
        human_score = human_score_by_box[_box_key(human["box"])]
        hover_boxes.append(
            {
                **_percent_box(human["box"], width, height),
                "label": f"Rider: {status}",
                "kind": "rider-safe" if rider["wearing_helmet"] else "rider-danger",
                "metrics": _human_threshold_metrics(
                    human_score,
                    rider_scale,
                    helmet_scale,
                    rider_score_threshold,
                    helmet_score_threshold,
                ),
            }
        )

    return hover_boxes

def process_image(
    path: Path,
    confidence: float,
    rider_score_threshold: float,
    helmet_score_threshold: float,
) -> dict[str, Any]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Could not read uploaded image.")

    annotated, analysis = run_frame(
        image,
        confidence,
        rider_score_threshold,
        helmet_score_threshold,
    )
    height, width = image.shape[:2]
    clear_previous_outputs(path)
    output_name = f"{path.stem}_detected.jpg"
    output_path = OUTPUT_DIR / output_name
    cv2.imwrite(str(output_path), annotated)

    return {
        "kind": "image",
        "input_name": path.name,
        "confidence": confidence,
        "rider_score_threshold": rider_score_threshold,
        "helmet_score_threshold": helmet_score_threshold,
        "output_url": output_url(output_name, output_path),
        "summary": summarize_analysis(analysis),
        "riders": analysis["riders"],
        "hover_boxes": hover_boxes_for_analysis(analysis, width, height),
    }


def empty_summary() -> dict[str, Any]:
    return {
        "detection_count": 0,
        "riders": 0,
        "with_helmet": 0,
        "without_helmet": 0,
        "helmets": 0,
        "humans": 0,
        "motorcycles": 0,
        "compliance_percent": 0.0,
        "risk_level": "Low",
        "average_confidence": 0.0,
        "lowest_confidence": 0.0,
    }


def merge_video_summary(current: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    frame_summary = summarize_analysis(analysis)
    merged = current.copy()
    for key in ("detection_count", "riders", "with_helmet", "without_helmet", "helmets", "humans", "motorcycles"):
        merged[key] = max(merged[key], frame_summary[key])
    merged["risk_level"] = "High" if merged["without_helmet"] else "Low"
    merged["compliance_percent"] = (
        round((merged["with_helmet"] / merged["riders"]) * 100, 1)
        if merged["riders"]
        else 0.0
    )
    merged["average_confidence"] = max(
        merged["average_confidence"],
        frame_summary["average_confidence"],
    )
    if frame_summary["lowest_confidence"]:
        if merged["lowest_confidence"]:
            merged["lowest_confidence"] = min(
                merged["lowest_confidence"],
                frame_summary["lowest_confidence"],
            )
        else:
            merged["lowest_confidence"] = frame_summary["lowest_confidence"]
    return merged


def video_slowdown_factor() -> float:
    return min(10.0, max(1.0, env_float("VIDEO_SLOWDOWN_FACTOR", DEFAULT_VIDEO_SLOWDOWN_FACTOR)))


def encode_browser_video(source_path: Path, output_path: Path, slowdown_factor: float = 1.0) -> None:
    filters = ["scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    if slowdown_factor > 1.0:
        filters.insert(0, f"setpts={slowdown_factor:.6f}*PTS")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-an",
        "-vf",
        ",".join(filters),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip().splitlines()
        detail = message[-1] if message else "unknown ffmpeg error"
        raise ValueError(f"Could not encode browser-playable video: {detail}")


def process_video(
    path: Path,
    confidence: float,
    rider_score_threshold: float,
    helmet_score_threshold: float,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Could not read uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Uploaded video has an invalid frame size.")

    clear_previous_outputs(path)
    slowdown_factor = video_slowdown_factor()
    output_name = f"{path.stem}_detected_slow.mp4"
    raw_output_name = f"{path.stem}_detected_raw.mp4"
    output_path = OUTPUT_DIR / output_name
    raw_output_path = OUTPUT_DIR / raw_output_name
    writer = cv2.VideoWriter(
        str(raw_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError("Could not create the annotated video output.")

    summary = empty_summary()
    representative_riders: list[dict[str, Any]] = []
    representative_score = -1
    frame_count = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            annotated, analysis = run_frame(
                frame,
                confidence,
                rider_score_threshold,
                helmet_score_threshold,
                show_labels=True,
            )
            writer.write(annotated)
            summary = merge_video_summary(summary, analysis)

            score = analysis["counts"]["without_helmet"] * 100 + analysis["counts"]["riders"]
            if score > representative_score:
                representative_score = score
                representative_riders = analysis["riders"]

            frame_count += 1
    finally:
        capture.release()
        writer.release()

    if frame_count == 0:
        raw_output_path.unlink(missing_ok=True)
        raise ValueError("Uploaded video did not contain readable frames.")

    encode_browser_video(raw_output_path, output_path, slowdown_factor)
    raw_output_path.unlink(missing_ok=True)

    return {
        "kind": "video",
        "input_name": path.name,
        "confidence": confidence,
        "rider_score_threshold": rider_score_threshold,
        "helmet_score_threshold": helmet_score_threshold,
        "output_url": output_url(output_name, output_path),
        "summary": summary,
        "riders": representative_riders,
        "frame_count": frame_count,
        "slowdown_factor": slowdown_factor,
    }


def process_job(job: ProcessingJob) -> dict[str, Any]:
    if job.kind == "image":
        return process_image(
            job.input_path,
            job.confidence,
            job.rider_score_threshold,
            job.helmet_score_threshold,
        )
    return process_video(
        job.input_path,
        job.confidence,
        job.rider_score_threshold,
        job.helmet_score_threshold,
    )


def worker_loop() -> None:
    global _active_job_id
    while True:
        job_id = _job_queue.get()
        try:
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job is None:
                    continue
                job.status = "processing"
                _active_job_id = job_id

            try:
                result = process_job(job)
            except Exception as exc:
                with _jobs_lock:
                    job.status = "failed"
                    job.error = str(exc)
            else:
                with _jobs_lock:
                    job.status = "done"
                    job.result = result
        finally:
            with _jobs_lock:
                if _active_job_id == job_id:
                    _active_job_id = None
            _job_queue.task_done()


def ensure_worker_started() -> None:
    global _worker_started
    with _jobs_lock:
        if _worker_started:
            return
        worker = threading.Thread(target=worker_loop, name="media-processing-worker", daemon=True)
        worker.start()
        _worker_started = True


def enqueue_job(
    input_path: Path,
    kind: str,
    confidence: float,
    rider_score_threshold: float,
    helmet_score_threshold: float,
) -> ProcessingJob:
    ensure_worker_started()
    job = ProcessingJob(
        id=uuid.uuid4().hex,
        input_path=input_path,
        kind=kind,
        confidence=confidence,
        rider_score_threshold=rider_score_threshold,
        helmet_score_threshold=helmet_score_threshold,
    )
    with _jobs_lock:
        _jobs[job.id] = job
    _job_queue.put(job.id)
    return job


def queued_job_ids() -> list[str]:
    return [job_id for job_id, job in _jobs.items() if job.status == "queued"]


def job_snapshot(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        queued_ids = queued_job_ids()
        queue_position = queued_ids.index(job_id) + 1 if job_id in queued_ids else 0
        return {
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "queue_position": queue_position,
            "jobs_ahead": max(0, queue_position - 1),
            "is_active": _active_job_id == job_id,
            "error": job.error,
            "result": job.result,
        }


def input_path_from_request() -> Path:
    existing_file = request.form.get("existing_file", "")
    uploaded_file = request.files.get("file")

    if existing_file:
        return existing_upload_path(existing_file)

    if not uploaded_file or uploaded_file.filename == "":
        raise ValueError("Choose an image or video file.")

    if not allowed_file(uploaded_file.filename):
        raise ValueError("Unsupported file type. Choose an image or video file.")

    filename = f"{uuid.uuid4().hex}_{secure_filename(uploaded_file.filename)}"
    input_path = UPLOAD_DIR / filename
    uploaded_file.save(input_path)
    return input_path


@app.route("/", methods=["GET", "POST"])
def index():
    model_ready = MODEL_PATH.exists()

    if request.method == "POST":
        confidence = confidence_from_form()
        rider_score_threshold = threshold_from_form("rider_score_threshold", RIDER_SCORE_THRESHOLD)
        helmet_score_threshold = threshold_from_form("helmet_score_threshold", HELMET_SCORE_THRESHOLD)
        try:
            input_path = input_path_from_request()
            kind = media_type(input_path)
            job = enqueue_job(
                input_path,
                kind,
                confidence,
                rider_score_threshold,
                helmet_score_threshold,
            )
        except Exception as exc:
            flash(str(exc))
            return redirect(url_for("index"))

        return redirect(url_for("job_page", job_id=job.id))

    return render_template("index.html", result=None, job=None, model_path=MODEL_PATH, model_ready=model_ready)


@app.route("/jobs/<job_id>")
def job_page(job_id: str):
    snapshot = job_snapshot(job_id)
    if snapshot is None:
        flash("Queued job was not found.")
        return redirect(url_for("index"))

    return render_template(
        "index.html",
        result=snapshot["result"] if snapshot["status"] == "done" else None,
        job=snapshot,
        model_path=MODEL_PATH,
        model_ready=MODEL_PATH.exists(),
    )


@app.route("/jobs/<job_id>/status")
def job_status(job_id: str):
    snapshot = job_snapshot(job_id)
    if snapshot is None:
        return jsonify({"status": "missing", "error": "Queued job was not found."}), 404
    return jsonify(
        {
            "status": snapshot["status"],
            "queue_position": snapshot["queue_position"],
            "jobs_ahead": snapshot["jobs_ahead"],
            "is_active": snapshot["is_active"],
            "error": snapshot["error"],
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5001")), debug=True)
