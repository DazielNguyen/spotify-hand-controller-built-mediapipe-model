# Models Module

## Mục đích

Module `models/` chịu trách nhiệm định nghĩa và xây dựng các mô hình deep learning cho pipeline nhận diện bàn tay. Project này tách biệt hai mô hình chính: Hand Detector và Landmark Model.

## Chức năng chính

### 1. Hand Detector (`hand_detector/`)
- **model.py**: Xây dựng mô hình phát hiện vị trí bàn tay trong ảnh
  - Input: Ảnh webcam (RGB)
  - Output: Bounding box của bàn tay hoặc heatmap
  - Mục đích: Crop và định vị vùng chứa bàn tay trước khi detect landmarks
  - Architecture đề xuất: Lightweight CNN (MobileNet backbone)

### 2. Landmark Model (`landmark_model/`)
- **model.py**: Xây dựng mô hình ước lượng 21 điểm landmark trên bàn tay
  - Input: Ảnh bàn tay đã được crop (256x256)
  - Output: 21 landmarks × (2 cho 2D hoặc 3 cho 3D) = 42 hoặc 63 giá trị
  - Mục đích: Ước lượng chính xác vị trí các khớp tay
  - Architecture đề xuất: Regression head trên feature extractor

## Cấu trúc

```
models/
├── hand_detector/
│   ├── __init__.py
│   └── model.py          # Hand detector model builder
└── landmark_model/
    ├── __init__.py
    └── model.py          # Landmark prediction model builder
```

## Sử dụng

```python
from models.hand_detector.model import build_hand_detector
from models.landmark_model.model import build_landmark_model

# Build hand detector
hand_detector = build_hand_detector(input_shape=(256, 256, 3))

# Build landmark model
landmark_model = build_landmark_model(input_shape=(256, 256, 3), num_landmarks=21)
```

## Pipeline hoạt động

```
Input Image (Webcam)
       │
       ▼
┌─────────────────┐
│  Hand Detector │ ───> Crop bàn tay
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Landmark Model │ ───> 21 landmarks (42/63 values)
└─────────────────┘
       │
       ▼
   Gesture Classification
```

## Ghi chú

- Module đang trong giai đoạn placeholder (chưa implement)
- Mục tiêu: Không dùng MediaPipe, tự xây dựng models
- Ưu tiên: Lightweight models để chạy realtime trên CPU
- Dataset sẽ cần thu thập và gán nhãn thủ công
