# Real-Time Inference System

## 1. Overview

The real-time inference system processes webcam frames and performs gesture-based music control.

Pipeline:

```

Webcam → Hand Detection → Hand Crop → Landmark Prediction → Gesture Recognition → Decision Layer → Music Action

```

---

# 2. Webcam Input

Library:

```

OpenCV

```

Capture frames:

```

cv2.VideoCapture(0)

```

Resolution:

```

640 x 480

```

For faster inference:

```

320 x 240

```

---

# 3. Inference Pipeline

Steps per frame:

1. Capture webcam frame
2. Detect hand bounding box
3. Crop hand region
4. Resize to model input
5. Run landmark prediction
6. Recognize gesture
7. Apply confidence threshold + debounce + cooldown
8. Trigger system action

---

# 4. Inference Pseudocode

```

while True:

```

frame = camera.read()

bbox = detect_hand(frame)

if bbox:

    hand = crop(frame, bbox)

    landmarks = landmark_model(hand)

    gesture = gesture_classifier(landmarks)

    execute_action(gesture)

```

```

---

# 5. Gesture Recognition

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

Classifier options:

```

MLP
SVM
RandomForest

```

---

# 6. MacOS Music Control

System commands are triggered using:

```

pyautogui

```

Example:

```

play/pause
next track
previous track
volume up
volume down

```

Example code:

```

pyautogui.press("playpause")

```

---

# 7. Command Safety Rules

To make the app reliable for daily use:

1. Use confidence threshold before accepting gesture
2. Require N consecutive frames for debounce
3. Add per-command cooldown window
4. If gesture is unstable, classify as `unknown` and do nothing

---

# 8. Performance Optimization

## Frame Skipping

Run detection every few frames.

```

detect every 3 frames

```

---

## Hand Tracking

Reuse previous bounding box using tracking.

Example:

```

optical flow

```

---

## Lower Resolution

```

640x480 → 320x240

```

---

## TensorFlow Lite

Convert model for faster inference.

Benefits:

```

faster CPU inference
smaller model
lower latency

```

---

# 9. Expected Performance

Target latency:

| Stage          | Latency |
| -------------- | ------- |
| Hand Detection | 8 ms    |
| Landmark Model | 10 ms   |
| Gesture Model  | 1 ms    |

Total:

```

~19 ms

```

Expected FPS:

```

20–30 FPS

```

---

# 10. Future Improvements

Possible enhancements:

```

multi-hand detection
temporal gesture recognition
sequence models (LSTM)
GPU acceleration
mobile deployment

```

```

---

💡 **Gợi ý thêm cho repo của bạn (rất đáng làm):**

Nếu bạn muốn project **trông giống AI research project thật sự**, tôi khuyên thêm 2 file nữa:

```

docs/
├── training_pipeline.md
├── experiment_results.md

```

Vì **AI recruiter rất thích thấy experiment tracking**.


```
