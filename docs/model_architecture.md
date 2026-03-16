# Model Architecture

## 1. Overview

This project uses a two-stage deep learning architecture:

1. Hand Detection Model
2. Hand Landmark Estimation Model

This design follows the common pipeline used in modern hand pose estimation systems.

```

Frame → Hand Detection → Crop → Landmark Prediction

```

The separation improves performance and reduces computational cost.

---

# 2. Hand Detection Model

## Objective

Detect the bounding box of the hand in an image frame.

Without this stage, the landmark model would need to process the entire image, which is inefficient.

---

## Model Architecture

Backbone:

```

MobileNetV3 Small

```

Chosen because:

- lightweight
- optimized for mobile and CPU inference
- good performance per parameter

---

## Input

```

256 x 256 RGB image

```

---

## Output

```

Bounding Box

[x_center, y_center, width, height]

```

Bounding box values are normalized between:

```

0 – 1

```

---

## Loss Function

Bounding box regression loss:

```

Mean Squared Error

```

or

```

Smooth L1 Loss

```

---

# 3. Hand Landmark Model

## Objective

Predict **21 hand keypoints** from the cropped hand image.

Each keypoint represents a joint in the hand.

---

## Input

```

128 x 128 RGB hand image

```

---

## Output

```

21 keypoints

```

Each keypoint has coordinates:

```

(x, y)

```

Total outputs:

```

21 × 2 = 42

```

---

## Model Structure

```

Input Image (128x128)
│
▼
Conv Layer (3x3)
│
▼
MobileNetV3 Backbone
│
▼
Global Average Pooling
│
▼
Dense Layer (256)
│
▼
Dense Layer (128)
│
▼
Output Layer (42)

```

---

## Parameter Target

```

< 3 million parameters

```

This constraint ensures the model runs efficiently on CPU.

---

## Loss Function

```

Mean Squared Error (MSE)

```

```

Loss = mean((predicted_keypoints - ground_truth_keypoints)^2)

```

---

# 4. Keypoint Coordinate System

All keypoints are normalized relative to image size.

Example:

```

x = x_pixel / image_width
y = y_pixel / image_height

```

Range:

```

0 – 1

```

---

# 5. Training Strategy

Training procedure:

1. Train hand detection model
2. Use bounding boxes to crop hands
3. Train landmark model using cropped images

---

# 6. Model Optimization

For real-time performance, the trained model will be converted to:

```

TensorFlow Lite

```

Benefits:

- smaller model size
- faster CPU inference
- lower latency

```

```
