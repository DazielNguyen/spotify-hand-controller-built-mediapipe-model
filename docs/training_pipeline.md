# Training Pipeline

## 1. Overview

This document describes the training workflow used to train the models in the project.

Two models are trained:

1. Hand Detection Model
2. Hand Landmark Model

The training pipeline ensures reproducibility, scalability, and consistent experiment tracking.

---

# 2. Training Workflow

Overall training process:

```

Dataset
│
▼
Data Preprocessing
│
▼
Data Augmentation
│
▼
Dataset Loader
│
▼
Model Training
│
▼
Model Evaluation
│
▼
Model Export

```

---

# 3. Environment Setup

Recommended environment:

Python version:

```

Python 3.10

```

Deep learning framework:

```

TensorFlow 2.x

```

Key dependencies:

```

tensorflow
opencv-python
numpy
scikit-learn
matplotlib

```

---

# 4. Data Loading

Dataset is loaded using:

```

tf.data.Dataset

```

Advantages:

- efficient data streaming
- built-in parallelization
- automatic batching

Example pipeline:

```

tf.data.Dataset
→ map(preprocess)
→ map(augment)
→ batch
→ prefetch

```

---

# 5. Data Augmentation

To prevent overfitting and improve generalization.

Augmentations applied during training:

```

Random rotation
Random scaling
Random brightness
Horizontal flip
Gaussian noise

```

Typical ranges:

```

rotation: ±40°
scale: 0.8 – 1.2
brightness: ±20%

```

---

# 6. Landmark Model Training

## Input

```

128 x 128 RGB hand image

```

## Output

```

42 values

```

Representing:

```

21 keypoints (x,y)

```

---

## Loss Function

Mean Squared Error:

```

Loss = mean((y_pred - y_true)^2)

```

---

## Optimizer

```

Adam

```

Learning rate:

```

0.001

```

---

## Batch Size

Typical value:

```

32

```

---

## Training Epochs

Typical range:

```

30 – 100 epochs

```

Training stops when validation loss stops improving.

---

# 7. Learning Rate Scheduling

Learning rate can be reduced when validation loss plateaus.

Example:

```

ReduceLROnPlateau

```

Parameters:

```

factor = 0.5
patience = 5

```

---

# 8. Model Checkpointing

Save the best model during training.

Example:

```

ModelCheckpoint

```

Criteria:

```

lowest validation loss

```

Saved file:

```

best_landmark_model.h5

```

---

# 9. Evaluation Metrics

Main metric:

```

Mean Squared Error (MSE)

```

Additional evaluation metrics:

```

Mean Absolute Error (MAE)

```

Visualization metric:

```

Keypoint overlay accuracy

```

---

# 10. Training Visualization

Training metrics are visualized using:

```

TensorBoard

```

Metrics tracked:

```

training loss
validation loss
learning rate

```

---

# 11. Model Export

After training, models are exported.

Format:

```

TensorFlow SavedModel

```

Then converted to:

```

TensorFlow Lite

```

Purpose:

```

faster inference
smaller model size

```

---

# 12. Final Model Files

Example output files:

```

models/
├── hand_detector_model.h5
├── landmark_model.h5
└── landmark_model.tflite

```

```

```
