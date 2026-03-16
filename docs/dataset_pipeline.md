# Dataset Pipeline

## 1. Overview

The dataset pipeline prepares training data for the hand detection and hand landmark models.

The pipeline includes:

- dataset loading
- annotation parsing
- preprocessing
- data augmentation
- batching

---

# 2. Dataset Sources

Recommended datasets:

## FreiHAND

Features:

```

130K images
21 hand keypoints
RGB images

```

---

## OneHand10K

Features:

```

10K annotated hand images
21 keypoints

```

---

# 3. Annotation Format

Each image has:

```

21 keypoints

```

Each keypoint:

```

(x, y)

```

Example format:

```

{
"image": "img_001.jpg",
"keypoints": [
[x1,y1],
[x2,y2],
...
[x21,y21]
]
}

```

---

# 4. Data Preprocessing

Steps:

1. Load image
2. Extract hand bounding box
3. Crop hand region
4. Resize image
5. Normalize pixel values

---

## Image Resize

```

128 x 128

```

Used as input to the landmark model.

---

## Pixel Normalization

```

pixel_value / 255

```

Range:

```

0 – 1

```

---

# 5. Data Augmentation

To improve model robustness.

Techniques:

```

Random rotation
Random scaling
Brightness adjustment
Horizontal flip
Gaussian noise
Random occlusion

```

Example ranges:

```

rotation: ±40 degrees
scale: 0.8 – 1.2
brightness: ±20%

```

---

# 6. Keypoint Transformation

When augmentation is applied, keypoints must be transformed accordingly.

Example:

Rotation transformation:

```

[x',y'] = R * [x,y]

```

Where:

```

R = rotation matrix

```

---

# 7. Dataset Loader

Data is loaded using TensorFlow dataset pipeline.

Example:

```

tf.data.Dataset

```

Pipeline:

```

load images
→ preprocess
→ augment
→ batch
→ training

```

---

# 8. Batching

Typical batch size:

```

32

```

Depending on available memory.

---

# 9. Training Split

Dataset should be split into:

```

train: 80%
validation: 10%
test: 10%

```

---

# 10. Dataset Output

Final output for training:

```

image tensor: (128,128,3)
keypoints: (42)

```

Example:

```

X → hand image
y → 42 landmark coordinates

```

```

```
