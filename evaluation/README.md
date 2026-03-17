# Evaluation Module

## Mục đích

Module `evaluation/` chịu trách nhiệm về việc đánh giá hiệu suất của các mô hình đã huấn luyện và hệ thống điều khiển ứng dụng.

## Chức năng chính

### 1. Landmark Evaluation (`eval_landmark.py`)
- **eval_landmark.py**: Đánh giá mô hình Landmark Model
  - Tính toán các metrics: MSE, MAE, PCK (Percentage of Correct Keypoints)
  - So sánh landmarks dự đoán với ground truth
  - Đánh giá trên tập test riêng biệt
  - Phân tích lỗi theo từng landmark

### 2. App Control Evaluation (`eval_app_control.py`)
- **eval_app_control.py**: Đánh giá hệ thống điều khiển ứng dụng
  - Test các gesture commands (play/pause, next, previous, volume)
  - Đo latency từ gesture đến action
  - Đánh giá false trigger rate
  - Tính command success rate

## Cấu trúc

```
evaluation/
├── __init__.py
├── eval_landmark.py       # Landmark model evaluation
└── eval_app_control.py   # App control evaluation
```

## Sử dụng

```bash
# Đánh giá landmark model
python evaluation/eval_landmark.py \
    --model_path models/landmark_model.h5 \
    --test_data data/test/

# Đánh giá app control
python evaluation/eval_app_control.py \
    --gesture_model models/gesture_classifier.h5 \
    --num_tests 100
```

## Metrics

### Landmark Evaluation
| Metric | Mô tả |
|--------|-------|
| MSE | Mean Squared Error |
| MAE | Mean Absolute Error |
| PCK@0.5 | % landmarks within 50% of bounding box |
| AUC | Area Under Curve |

### App Control Evaluation
| Metric | Mô tả |
|--------|-------|
| Success Rate | % commands executed successfully |
| False Trigger Rate | % accidental triggers |
| Latency | Time from gesture to action |
| Precision | True Positive / (True Positive + False Positive) |
| Recall | True Positive / (True Positive + False Negative) |

## Ghi chú

- Module đang trong giai đoạn placeholder
- Cần có trained model và test data trước khi đánh giá
- Nên chạy evaluation sau mỗi epoch training để theo dõi tiến độ
- App control evaluation cần test trực tiếp với Spotify/Apple Music
