# Experiment Results

## 1. Overview

This document summarizes the experiments conducted during model development.

Experiments focus on:

- model architecture
- training stability
- inference performance
- real-time capability

---

# 2. Experiment Setup

Hardware:

```

CUDA
```

Software:

```

TensorFlow 2.x
OpenCV
Python 3.10

```

Dataset:

```

FreiHAND
OneHand10K

```

---

# 3. Landmark Model Experiments

## Experiment 1

Model:

```

Simple CNN

```

Architecture:

```

Conv → Conv → Dense

```

Parameters:

```

~5M

```

Results:

| Metric          | Value  |
| --------------- | ------ |
| Training Loss   | 0.0035 |
| Validation Loss | 0.0042 |
| FPS             | 12     |

Observation:

```

Model accuracy acceptable but inference too slow.

```

---

## Experiment 2

Model:

```

MobileNetV3 backbone

```

Parameters:

```

~2.5M

```

Results:

| Metric          | Value  |
| --------------- | ------ |
| Training Loss   | 0.0028 |
| Validation Loss | 0.0034 |
| FPS             | 28     |

Observation:

```

Better balance between accuracy and speed.

```

---

# 4. Gesture Recognition Experiments

Input features:

```

42 keypoint values

```

Tested classifiers:

```

MLP
SVM
RandomForest

```

Results:

| Model        | Accuracy |
| ------------ | -------- |
| MLP          | 92%      |
| SVM          | 90%      |
| RandomForest | 88%      |

Best model:

```

MLP

```

---

# 5. Real-Time Performance

Measured latency per stage:

| Component              | Latency |
| ---------------------- | ------- |
| Hand Detection         | 8 ms    |
| Landmark Prediction    | 10 ms   |
| Gesture Classification | 1 ms    |

Total latency:

```

~19 ms

```

Equivalent FPS:

```

~50 FPS theoretical

```

Observed FPS:

```

25–30 FPS

```

---

# 6. Visual Results

Example output:

```

webcam frame
│
▼
hand detected
│
▼
21 keypoints drawn on hand

```

Example landmarks:

```

wrist
thumb joints
index joints
middle joints
ring joints
pinky joints

```

---

# 7. System Demonstration

The final system can perform gesture-based control of macOS.

Supported commands:

| Gesture     | Action      |
| ----------- | ----------- |
| Open hand   | Play        |
| Fist        | Pause       |
| Two fingers | Next track  |
| Thumb up    | Volume up   |
| Thumb down  | Volume down |

---

# 8. Limitations

Current system limitations:

```

single hand only
sensitive to heavy occlusion
performance drops in low light

```

---

# 9. Future Improvements

Potential improvements:

```

multi-hand tracking
temporal gesture recognition
LSTM gesture models
GPU acceleration
mobile deployment

```

```




```
