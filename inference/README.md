# Inference Module

## Mục đích

Module `inference/` chịu trách nhiệm về việc chạy inference (dự đoán) với các mô hình đã huấn luyện. Bao gồm inference cho ảnh đơn và video/webcam.

## Chức năng chính

### 1. Single Image Inference (`inference_image.py`)
- **inference_image.py**: Chạy inference trên một ảnh đơn lẻ
  - Load ảnh từ file
  - Preprocess ảnh
  - Run hand detector để tìm vùng bàn tay
  - Run landmark model để lấy 21 landmarks
  - Visualize kết quả (vẽ landmarks lên ảnh)
  - Lưu ảnh kết quả vào `outputs/`

### 2. Webcam Inference (`webcam_inference.py`)
- **webcam_inference.py**: Chạy inference realtime từ webcam
  - Capture frame từ webcam
  - Run detection và landmark extraction
  - Real-time visualization với overlay landmarks
  - Hiển thị FPS performance

## Cấu trúc

```
inference/
├── __init__.py
├── inference_image.py    # Single image inference
└── webcam_inference.py  # Real-time webcam inference
```

## Sử dụng

```bash
# Inference trên ảnh đơn
python inference/inference_image.py \
    --image_path inputs/test.jpg \
    --model_path models/landmark_model.h5 \
    --output_path outputs/result.jpg

# Inference từ webcam
python inference/webcam_inference.py \
    --model_path models/landmark_model.h5
```

## Pipeline Inference

```
┌──────────────────────────────────────────────────┐
│              Inference Pipeline                  │
├──────────────────────────────────────────────────┤
│ 1. Load & preprocess image                       │
│ 2. Hand Detection (bounding box)                │
│ 3. Crop hand region                             │
│ 4. Landmark Prediction (21 points)             │
│ 5. Post-process landmarks                       │
│ 6. Visualization (draw on image)                │
│ 7. Output (save/image display)                  │
└──────────────────────────────────────────────────┘
```

## Output

- Ảnh với các điểm landmark được vẽ lên
- Tọa độ 21 landmarks (format: x1, y1, x2, y2, ...)
- Thông tin thêm: bounding box của bàn tay, confidence score

## Ghi chú

- Module đang trong giai đoạn placeholder
- Cần train model trước khi chạy inference
- Webcam inference yêu cầu webcam được kết nối và cấp quyền
- Để tối ưu FPS, có thể crop frame thay vì process toàn bộ
