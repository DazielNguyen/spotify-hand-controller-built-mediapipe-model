# 7-Day Plan: Dataset Pipeline

## Ngày 1: Chuẩn bị & Dataset Loading
- [ ] Nghiên cứu và tải FreiHAND dataset (130K images, 21 keypoints)
- [ ] Nghiên cứu và tải OneHand10K dataset (10K images)
- [ ] Thiết lập cấu trúc thư mục dataset
- [ ] Viết script download và giải nén datasets

## Ngày 2: Annotation Parsing
- [ ] Thiết kế định dạng annotation JSON schema
- [ ] Viết parser cho FreiHAND format
- [ ] Viết parser cho OneHand10K format
- [ ] Tạo unified annotation format
- [ ] Validate annotations

## Ngày 3: Data Preprocessing
- [ ] Implement image loading (OpenCV/PIL)
- [ ] Implement bounding box extraction từ keypoints
- [ ] Implement hand region cropping
- [ ] Implement resize về 128x128
- [ ] Implement pixel normalization (0-1 range)

## Ngày 4: Data Augmentation
- [ ] Implement random rotation (±40 degrees)
- [ ] Implement random scaling (0.8-1.2)
- [ ] Implement brightness adjustment (±20%)
- [ ] Implement horizontal flip
- [ ] Implement Gaussian noise
- [ ] Implement random occlusion

## Ngày 5: Keypoint Transformation
- [ ] Implement rotation transformation cho keypoints
- [ ] Implement scaling transformation cho keypoints
- [ ] Implement flip transformation cho keypoints
- [ ] Test transformation consistency giữa image và keypoints
- [ ] Validate transformed keypoints

## Ngày 6: Dataset Loader & Batching
- [ ] Implement tf.data.Dataset pipeline
- [ ] Implement preprocessing pipeline
- [ ] Implement augmentation pipeline
- [ ] Implement batching (batch_size=32)
- [ ] Optimize data loading performance

## Ngày 7: Training Split & Output
- [ ] Implement train/val/test split (80/10/10)
- [ ] Verify output format: image tensor (128,128,3), keypoints (42)
- [ ] Create data generator utility
- [ ] Test full pipeline end-to-end
- [ ] Document usage và examples

---

## Dependencies
- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Pandas
- Pillow

## Output
- `dataset/download_datasets.py` - Script tải datasets
- `dataset/parsers/` - Annotation parsers
- `dataset/preprocessing/` - Preprocessing utilities
- `dataset/augmentation/` - Augmentation functions
- `dataset/loader/` - Dataset loader implementation
- `dataset/utils/` - Helper functions
- `dataset/test_pipeline.py` - Test end-to-end pipeline
