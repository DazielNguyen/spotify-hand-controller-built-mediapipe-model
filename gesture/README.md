# Gesture Classification Module

## Mục đích

Module `gesture/` chịu trách nhiệm về việc phân loại các gesture (cử chỉ tay) từ 21 landmarks đã được ước lượng. Module này chuyển đổi tọa độ landmarks thành các nhãn gesture có ý nghĩa.

## Chức năng chính

### 1. Gesture Classifier (`classifier.py`)
- **classifier.py**: Class phân loại gesture từ landmarks
  - Input: 21 landmarks (tọa độ x, y hoặc x, y, z)
  - Output: Nhãn gesture (string)
  - Các gesture được hỗ trợ:
    - `open_palm`: Bàn tay mở
    - `fist`: Nắm tay
    - `peace`: Hai ngón (V sign)
    - `thumbs_up`: Ngón cái chỉ lên
    - `pointing`: Chỉ tay
    - `swipe_left`: Vuốt sang trái
    - `swipe_right`: Vuốt sang phải

## Cấu trúc

```
gesture/
└── classifier.py         # Gesture classification logic
```

## Sử dụng

```python
from gesture.classifier import GestureClassifier

# Khởi tạo classifier
classifier = GestureClassifier()

# Phân loại gesture từ landmarks
landmarks = [x1, y1, x2, y2, ..., x21, y21]  # 42 values
gesture_label = classifier.predict(landmarks)

print(f"Detected gesture: {gesture_label}")  # e.g., "peace", "fist"
```

## Gesture Mapping

| Gesture | Action Mapping | Mô tả |
|---------|----------------|-------|
| `open_palm` | Play/Pause | Mở lòng bàn tay |
| `fist` | Stop/Close | Nắm tay lại |
| `peace` | Next Track | Hai ngón chữ V |
| `thumbs_up` | Volume Up | Ngón cái chỉ lên |
| `pointing` | Volume Down | Chỉ tay xuống |
| `swipe_left` | Previous Track | Vuốt sang trái |
| `swipe_right` | Next Track | Vuốt sang phải |

## Cách hoạt động

```
21 Landmarks (from Landmark Model)
         │
         ▼
┌─────────────────────┐
│  Feature Extraction │ ───> Tính khoảng cách, góc
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Rule-based / ML    │ ───> Classification
└─────────────────────┘
         │
         ▼
   Gesture Label
```

## Ghi chú

- Module đang trong giai đoạn placeholder
- Có thể implement bằng rule-based (if/else) hoặc ML classifier
- Rule-based: sử dụng khoảng cách giữa các ngón tay
- ML classifier: train classifier riêng với labeled data
- Nên thêm gesture smoothing/debounce để tránh false triggers
