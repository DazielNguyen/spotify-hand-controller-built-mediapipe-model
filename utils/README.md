# Utils Module

## Mục đích

Module `utils/` chứa các hàm tiện ích (utilities) được sử dụng chung bởi nhiều module khác trong project. Các hàm này hỗ trợ xử lý landmarks, visualization, và các tác vụ helper khác.

## Chức năng chính

### 1. Landmark Utilities (`landmarks.py`)
- **landmarks.py**: Các hàm xử lý và phân tích landmarks
  - `as_pairs()`: Chuyển đổi flat vector thành array (N, 2) hoặc (N, 3)
  - `normalize_relative_to_wrist()`: Chuẩn hóa landmarks theo vị trí cổ tay (index 0)
  - `finger_is_open()`: Kiểm tra ngón tay có mở hay không
  - `classify_simple_gesture()`: Phân loại gesture đơn giản (rule-based)
  - `HAND_CONNECTIONS`: Tuple định nghĩa kết nối giữa các landmarks để vẽ skeleton

## Cấu trúc

```
utils/
├── __init__.py
└── landmarks.py          # Landmark processing utilities
```

## Sử dụng

```python
from utils.landmarks import (
    as_pairs,
    normalize_relative_to_wrist,
    finger_is_open,
    classify_simple_gesture,
    HAND_CONNECTIONS
)

# Chuyển đổi flat array thành cặp tọa độ
landmarks_2d = as_pairs([x1, y1, x2, y2, ...])  # Shape: (21, 2)

# Chuẩn hóa theo cổ tay
normalized = normalize_relative_to_wrist(landmarks_2d)

# Kiểm tra ngón tay có mở không
is_open = finger_is_open(landmarks_2d, tip_index=8, pip_index=6)

# Phân loại gesture đơn giản
gesture = classify_simple_gesture(landmarks_2d)  # e.g., "open_palm", "fist"

# Vẽ skeleton
print(HAND_CONNECTIONS)  # ((0,1), (1,2), ...)
```

## Hand Connections (21 Landmarks)

```
MediaPipe Hand Landmarks Index:
         8           12          16          20
         |            |           |           |
         7----6----5  11---10---9  15---14---19
          \          / \          / \          /
           4--------3   2--------1   0---------13
          /                  \                /
         3                   18-------------17
        /                       \           /
       2                         18-------17
      /                             \    /
     1-----------0-------------------0   /
    /              \                  \ /
   5----------------18              18
```

| Index | Landmark | Mô tả |
|-------|----------|-------|
| 0 | WRIST | Cổ tay |
| 1-4 | THUMB | Ngón cái |
| 5-8 | INDEX_FINGER | Ngón trỏ |
| 9-12 | MIDDLE_FINGER | Ngón giữa |
| 13-16 | RING_FINGER | Ngón áp út |
| 17-20 | PINKY | Ngón út |

## Các hàm chi tiết

### `as_pairs(landmarks)`
Chuyển đổi flat landmark array thành dạng cặp tọa độ.

```python
# Input: [x1, y1, x2, y2, ...] hoặc [x1, y1, z1, x2, y2, z2, ...]
# Output: [[x1, y1], [x2, y2], ...] hoặc [[x1, y1, z1], ...]
```

### `normalize_relative_to_wrist(landmarks)`
Chuẩn hóa tất cả landmarks theo vị trí cổ tay (index 0).

### `finger_is_open(points, tip_index, pip_index)`
Kiểm tra ngón tay có mở hay không dựa trên vị trí tương đối của tip và PIP joint.

### `classify_simple_gesture(landmarks)`
Phân loại gesture đơn giản:
- `fist`: Tất cả ngón đóng
- `open_palm`: Tất cả ngón mở
- `peace`: Index và middle mở
- `thumbs_up`: Chỉ thumb mở
- `unknown`: Các trường hợp khác

## Ghi chú

- Đây là module đã có code thực (không phải placeholder)
- Các hàm này được sử dụng bởi `gesture/`, `inference/`, và các module khác
- Có thể mở rộng thêm các hàm visualization (vẽ landmarks)
