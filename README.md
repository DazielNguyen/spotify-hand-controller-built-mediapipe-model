# Spotify Hand Controller Built Without MediaPipe

TensorFlow-based hand gesture control project for macOS. The goal is to run a lightweight pipeline on webcam input, estimate 21 hand landmarks without MediaPipe, classify gestures, and map those gestures to real music-control actions such as play/pause, next track, previous track, and volume control.

## Status

This repository is currently organized as a structure-only scaffold:

- data loading and preprocessing helpers are in `data/`
- model builders are in `models/`
- training entrypoints are in `training/`
- inference entrypoints are in `inference/`
- evaluation helpers are in `evaluation/`
- gesture and macOS action logic remain in `gesture/` and `mac_control/`

The repository currently focuses on folder layout, module boundaries, and file placeholders. Model training, inference, evaluation, gesture logic, and macOS control are not implemented yet.

Final product target:

- realtime webcam-based hand gesture music controller
- command safety layer: threshold + debounce + cooldown
- practical UX with low false-trigger rate

## Project Structure

```text
.
├── data/
│   ├── dataset.py
│   └── transforms.py
├── docs/
├── evaluation/
│   └── eval_landmark.py
├── gesture/
│   └── classifier.py
├── inference/
│   ├── inference_image.py
│   └── webcam_inference.py
├── mac_control/
│   └── control.py
├── models/
│   ├── hand_detector/
│   │   ├── __init__.py
│   │   └── model.py
│   └── landmark_model/
│       ├── __init__.py
│       └── model.py
├── realtime/
│   └── webcam_inference.py
├── training/
│   └── train_landmark.py
├── utils/
│   └── landmarks.py
└── requirements.txt
```

## Requirements

- Python 3.10+
- macOS for system control actions
- Webcam for realtime inference

Install dependencies:

```bash
pip install -r requirements.txt
```

## Current Scope

- `training/` contains training entrypoint placeholders
- `inference/` contains inference entrypoint placeholders
- `evaluation/` contains evaluation entrypoint placeholders
- `models/` contains model-builder placeholders
- `gesture/` and `mac_control/` contain interface stubs only

## Next Implementation Targets

Suggested order for filling in the scaffold:

1. Implement dataset loading and preprocessing in `data/`
2. Implement model builders in `models/`
3. Implement training loop in `training/train_landmark.py`
4. Implement inference pipeline in `inference/`
5. Implement gesture mapping and macOS control logic
6. Add decision safety layer (threshold, debounce, cooldown)
7. Validate app-level metrics: command success, false trigger rate, command latency

## Notes

- This project intentionally does not depend on MediaPipe.
- Current code focuses only on repository structure and interface placeholders.
- See `docs/` for model architecture, training pipeline, realtime inference notes, and experiment planning.
