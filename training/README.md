# Training Module

## Mục đích

Module `training/` chịu trách nhiệm về việc huấn luyện các mô hình deep learning cho pipeline nhận diện bàn tay, bao gồm Hand Detector và Landmark Model.

## Chức năng chính

### 1. Landmark Model Training (`train_landmark.py`)
- **train_landmark.py**: Script huấn luyện mô hình dự đoán 21 landmarks từ ảnh bàn tay
  - Load dữ liệu từ `data/` module
  - Định nghĩa loss function (MSE/MAE cho regression)
  - Training loop với backpropagation
  - Validation sau mỗi epoch
  - Model checkpointing
  - Learning rate scheduling
  - TensorBoard logging

## Cấu trúc

```
training/
└── train_landmark.py    # Main training script
```

## Sử dụng

```bash
# Train landmark model
python training/train_landmark.py \
    --data_dir data/landmarks \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --checkpoint_dir checkpoints/
```

## Luồng huấn luyện

```
┌─────────────────────────────────────────┐
│           Training Pipeline             │
├─────────────────────────────────────────┤
│ 1. Load Dataset (from data/)            │
│ 2. Initialize Model (from models/)       │
│ 3. Define Loss (MSE/L1)                 │
│ 4. Setup Optimizer (Adam/SGD)           │
│ 5. Training Loop:                        │
│    - Forward pass                       │
│    - Compute loss                        │
│    - Backward pass                       │
│    - Update weights                     │
│ 6. Validate on validation set            │
│ 7. Save checkpoints                      │
└─────────────────────────────────────────┘
```

## Hyperparameters đề xuất

| Parameter | Giá trị đề xuất |
|-----------|-----------------|
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 100-200 |
| Optimizer | Adam |
| Loss Function | MSE |
| Image Size | 256x256 |

## Ghi chú

- Module đang trong giai đoạn placeholder
- Cần thu thập dataset trước khi train
- Nên implement data augmentation để tăng cường model robustness
- Cân nhắc sử dụng pre-trained backbone (MobileNet) để tăng tốc độ hội tụ
