# Helmet Detection Dashboard

This project is a small Flask app for checking motorcycle helmet compliance in images and videos using a YOLOv8 model.

It detects three classes:

- `helmet`
- `human`
- `motorcycle`

The app then does a second pass on top of the raw detections to decide:

- which humans are likely riders
- which riders are wearing helmets
- how many helmets, humans, motorcycles, and riders appear in the scene

## What The App Shows

- annotated image or video output
- counts for riders, with helmet, without helmet, helmets, humans, and motorcycles
- color legend for the bounding boxes
- hover tooltips on images with labels and score details
- adjustable `confidence`, `rider score threshold`, and `helmet score threshold`

## Requirements

- a trained YOLOv8 model file named `best.pt`
- `ffmpeg` installed and available on `PATH` if you want browser-playable processed video output
- a `test` folder with matching image and label files under `test/images` and `test/labels` to run `pre_annotate.py`, `review_annotations.py`, `eval_compliance.py`, `generate_all_comparisons.py`, and `generate_report_images.py`

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies.
3. Put your trained model at `models/best.pt`.
4. Start the Flask app.

```powershell
python3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

By default, the app runs at `http://127.0.0.1:5001`.

## How It Works

The model first predicts `helmet`, `human`, and `motorcycle` boxes. After that, the app applies extra logic to make the output more useful:

- duplicate detections are reduced with per-class non-maximum suppression
- each human is scored against nearby motorcycles to decide whether they are a rider
- each rider is matched against helmet detections using a head-region score
- helmet matches are assigned one-to-one so the same helmet is not counted twice

This is why the rider counts can differ from the raw `human` count.
