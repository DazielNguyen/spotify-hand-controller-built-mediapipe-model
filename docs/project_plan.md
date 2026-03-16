# Real-Time Hand Pose Estimation for MacOS Gesture Control (TensorFlow)

## 1. Project Overview

This project aims to build a **real-time hand pose estimation system** capable of detecting **21 hand keypoints** using **TensorFlow**, without relying on any external hand tracking frameworks.

The system will process webcam input, detect the hand, estimate hand landmarks, recognize gestures, and control system functions on macOS (such as play/pause music, next track, and volume control).

The entire pipeline must run **in real time (20–30 FPS)** on a MacBook CPU.

---

# 2. Objectives

## Primary Goals

- Detect human hand from webcam video
- Predict **21 hand keypoints**
- Recognize hand gestures
- Control macOS system functions
- Achieve **real-time performance**

## Technical Constraints

- Do NOT use MediaPipe
- Build models using **TensorFlow**
- System must run on **CPU in real-time**

---

# 3. System Architecture

## Overall Pipeline

```

Webcam Input
│
▼
Hand Detection Model
│
▼
Hand Crop & Preprocessing
│
▼
Hand Landmark Model (21 Keypoints)
│
▼
Gesture Recognition Model
│
▼
MacOS Control System

```

---

# 4. Module Design

## 4.1 Webcam Input

Capture frames from the system camera.

**Library**

```

OpenCV

```

**Input resolution**

```

640 x 480

```

To improve speed:

```

resize → 320 x 240

```

---

# 5. Hand Detection Module

## Purpose

Detect the bounding box of the hand in the image.

Running the landmark model on the full frame is computationally expensive.

Instead:

```

frame → detect hand → crop → landmark model

```

## Model Architecture

Backbone:

```

MobileNetV3 Small

```

Input:

```

256 x 256

```

Output:

```

Bounding Box (x, y, width, height)

```

## Training Dataset

Possible datasets:

- EgoHands
- Oxford Hand Dataset
- Custom collected dataset

---

# 6. Hand Landmark Estimation

## Purpose

Predict **21 hand keypoints**

Output format:

```

(21,2)

```

Total outputs:

```

42 values

```

Coordinates represent:

```

(x,y) position of each joint

```

---

## Hand Landmark Model Architecture

Input:

```

128 x 128 RGB image

```

Architecture:

```

Conv Layer
│
MobileNetV3 Backbone
│
Global Average Pooling
│
Dense Layer (256)
│
Dense Layer (128)
│
Output Layer (42)

```

Parameter target:

```

< 3M parameters

```

Loss Function:

```

Mean Squared Error (MSE)

```

---

# 7. Hand Keypoint Definition

The model predicts **21 landmarks**.

```

0  Wrist

Thumb
1  Thumb CMC
2  Thumb MCP
3  Thumb IP
4  Thumb Tip

Index Finger
5  MCP
6  PIP
7  DIP
8  Tip

Middle Finger
9
10
11
12

Ring Finger
13
14
15
16

Pinky Finger
17
18
19
20

```

---

# 8. Dataset

Recommended datasets for training:

### FreiHAND

Contains large number of hand pose images.

Key features:

```

130K images
21 keypoints
RGB images

```

---

### OneHand10K

Another labeled dataset for hand pose estimation.

```

10K images
21 keypoints

```

---

# 9. Data Preprocessing

Steps:

```

1. Load image
2. Crop hand region
3. Resize to 128x128
4. Normalize pixel values

```

Normalization:

```

pixel / 255

```

---

# 10. Data Augmentation

To improve robustness.

Augmentation techniques:

```

Random rotation
Random scaling
Horizontal flip
Brightness change
Gaussian noise
Random occlusion

```

Typical ranges:

```

rotation: ±40 degrees
scale: 0.8 – 1.2
brightness: ±20%

```

---

# 11. Gesture Recognition

Once the system predicts hand landmarks, we classify gestures.

Input:

```

21 keypoints

```

Feature vector:

```

42 values

```

Normalization:

```

relative to wrist

```

Example:

```

x_i - wrist_x
y_i - wrist_y

```

---

## Gesture Classifier

Simple models are sufficient:

```

MLP
SVM
RandomForest

```

Output:

```

Gesture label

```

Example gestures:

| Gesture     | Action         |
| ----------- | -------------- |
| Open hand   | Play music     |
| Fist        | Pause music    |
| Two fingers | Next track     |
| Swipe left  | Previous track |
| Thumb up    | Volume up      |
| Thumb down  | Volume down    |

---

# 12. MacOS Control Module

This module converts gestures into system actions.

Library:

```

pyautogui

```

Example commands:

```

play/pause
next track
volume up
volume down

```

Example code:

```python
import pyautogui

pyautogui.press("playpause")
pyautogui.press("nexttrack")
pyautogui.press("volumeup")
```

---

# 13. Real-Time Inference Pipeline

Main runtime loop:

```
while camera is running:

    capture frame

    detect hand

    crop hand

    predict landmarks

    recognize gesture

    trigger macOS action
```

Pseudo code:

```python
while True:

    frame = webcam.read()

    bbox = detect_hand(frame)

    if bbox:

        hand_crop = crop(frame, bbox)

        landmarks = landmark_model.predict(hand_crop)

        gesture = classify_gesture(landmarks)

        execute_action(gesture)
```

---

# 14. Real-Time Optimization

To maintain high FPS.

## Frame Skipping

```
run detection every 3 frames
```

---

## Tracking

Reuse previous bounding box.

```
optical flow
```

---

## Lower Resolution

```
640x480 → 320x240
```

---

## Model Quantization

Convert model to:

```
TensorFlow Lite
```

Benefits:

```
smaller size
faster inference
lower latency
```

---

# 15. Target Performance

Expected performance on MacBook CPU:

| Component      | Latency |
| -------------- | ------- |
| Hand Detection | ~8 ms   |
| Landmark Model | ~10 ms  |
| Gesture Model  | ~1 ms   |

Total latency:

```
~19 ms
```

Equivalent FPS:

```
~50 FPS theoretical
```

Real-world expected:

```
20–30 FPS
```

---

# 16. Project Structure

```
hand-gesture-control
│
├── dataset
│
├── models
│   ├── hand_detector
│   └── landmark_model
│
├── training
│   ├── train_detector.py
│   └── train_landmark.py
│
├── gesture
│   └── gesture_classifier.py
│
├── realtime
│   └── webcam_inference.py
│
├── mac_control
│   └── system_control.py
│
└── docs
    └── project_plan.md
```

---

# 17. Development Timeline

## Week 1 – Research

- Study hand pose estimation
- Explore datasets
- Define architecture

---

## Week 2 – Dataset Preparation

- Download dataset
- Convert annotation format
- Build data loader

---

## Week 3 – Hand Detection Model

- Train bounding box detector
- Evaluate accuracy

---

## Week 4 – Landmark Model

- Implement TensorFlow model
- Train keypoint predictor

---

## Week 5 – Gesture Recognition

- Extract features
- Train gesture classifier

---

## Week 6 – Real-Time System

- Integrate webcam pipeline
- Optimize inference

---

## Week 7 – MacOS Control

- Implement gesture actions
- Connect system commands

---

## Week 8 – Optimization

- TensorFlow Lite conversion
- Performance tuning

---

# 18. Expected Outcome

The final system will:

- Detect hand in webcam video
- Predict **21 hand landmarks**
- Recognize gestures
- Control macOS system functions
- Run **in real time**

---

# 19. Future Improvements

Possible upgrades:

- multi-hand tracking
- gesture sequence recognition
- temporal models (LSTM / Transformer)
- GPU acceleration
- mobile deployment

```


```
