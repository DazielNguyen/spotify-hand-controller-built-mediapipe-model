# Data Module

## Mục đích

Module `data/` chịu trách nhiệm về việc tải dữ liệu, xử lý và tiền xử lý dữ liệu huấn luyện mô hình nhận diện gesture từ bàn tay.

## Chức năng chính

### 1. Dataset Loading (`dataset.py`)
- **dataset.py**: Định nghĩa class `HandGestureDataset` để tải và quản lý dữ liệu training
  - Load ảnh từ webcam hoặc dataset có sẵn
  - Load nhãn gesture tương ứng (21 landmarks coordinates)
  - Hỗ trợ data augmentation
  - Data splitting (train/val/test)

### 2. Data Transforms (`transforms.py`)
- **transforms.py**: Các hàm transform và preprocessing cho dữ liệu
  - Resize ảnh về kích thước chuẩn (256x256 hoặc 224x224)
  - Normalize pixel values
  - Landmark normalization (chuyển đổi tọa độ tuyệt đối sang tọa độ tương đối)
  - Data augmentation: rotation, flip, brightness/contrast adjustment

## Cấu trúc dữ liệu

```
data/
├── __init__.py         # Export dataset classes
├── dataset.py          # Dataset loading logic
└── transforms.py       # Data preprocessing functions
```

## Sử dụng

```python
from data.dataset import HandGestureDataset
from data.transforms import get_train_transforms, get_val_transforms

# Load dataset
train_dataset = HandGestureDataset(
    data_dir="path/to/data",
    transform=get_train_transforms()
)
```

## Ghi chú

- Module này đang trong giai đoạn phát triển (placeholder)
- Cần implement chi tiết dataset loading từ nguồn dữ liệu cụ thể
- Dự kiến sẽ hỗ trợ nhiều định dạng ảnh (JPG, PNG, MP4 frames)
