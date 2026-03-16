# NGÀY 7: In-the-wild Inference & Báo cáo

**Mục tiêu:** Mang mô hình ra thực tế và gói gọn dự án.

---

## Sáng: Green Screen & Image Compositing (Section 10)

- [ ] Đọc Section 10 của paper — hiểu lý do cần augment background (FreiHAND chụp trên nền xanh, real-world thì không)
- [ ] Chuẩn bị background pool:
  - [ ] Tải 20-30 ảnh background đa dạng (rừng, đường phố, văn phòng, trong nhà) vào `dataset/backgrounds/`
  - [ ] Các ảnh nên có kích thước ≥ 224×224
- [ ] Load file Mask từ FreiHAND dataset: `training_segmentation.json` hoặc file mask `.png`
- [ ] Viết hàm `composite_background(image, mask, bg_pool)`:
  - [ ] Đọc mask → tạo alpha channel
  - [ ] Chọn background ngẫu nhiên từ `bg_pool`, resize về cùng kích thước ảnh
  - [ ] Alpha blending: `result = hand * mask + background * (1 - mask)`
- [ ] Tích hợp vào pipeline `tf.data.Dataset.map(...)` với xác suất 50% (giữ green screen 50% còn lại)
- [ ] Viết script debug: vẽ 1 batch 16 ảnh ra grid, lưu `outputs/bg_augmentation_grid.jpg`

> **CHECKPOINT 7.1:** Mở `bg_augmentation_grid.jpg`. Khoảng một nửa số ảnh phải thấy bàn tay nằm trên nền tự nhiên (rừng, đường, ...). Chất lượng compositing phải tự nhiên, không thấy viền xanh hay artifact.

---

## Chiều: Real-time / Inference Demo

- [ ] Tạo file `inference/webcam_inference.py` theo luồng sản phẩm thực tế:
  - [ ] Webcam input → hand detect/crop → landmark model → gesture classifier
  - [ ] Decision layer: threshold + debounce + cooldown
  - [ ] Action mapper: gesture -> command nhạc
  - [ ] macOS adapter: gửi media keys (`play_pause`, `next_track`, `previous_track`, `volume_up`, `volume_down`)
- [ ] Tạo chế độ `--dry-run` để in lệnh thay vì bắn phím thật khi đang debug
- [ ] Chuẩn bị bộ test gesture thực tế:
  - [ ] Nắm đấm -> play/pause
  - [ ] Tay xòe -> next track
  - [ ] Chữ V -> previous track
  - [ ] Thumbs up -> volume up
  - [ ] Thumbs down -> volume down
- [ ] Quay video demo 30-60 giây thể hiện luồng end-to-end
- [ ] Lưu artifact demo vào `outputs/demo/` (video + log lệnh)

> **CHECKPOINT 7.2:** Có video demo chạy realtime, thực hiện được ít nhất 3 lệnh nhạc khác nhau với độ trễ thấp và không bị trigger nhầm quá nhiều.

---

## Tối: Tổng kết Research & Dọn dẹp Code

### Cấu trúc thư mục

- [x] Dọn dẹp và tổ chức lại cấu trúc project:
  ```
  ├── data/
  │   ├── dataset.py          # dataset loading helpers
  │   └── transforms.py       # image and landmark preprocessing
  ├── docs/
  ├── models/
  │   ├── hand_detector/
  │   │   └── model.py
  │   └── landmark_model/
  │       └── model.py
  ├── training/
  │   └── train_landmark.py
  ├── evaluation/
  │   └── eval_landmark.py
  ├── utils/
  │   └── landmarks.py        # normalization and gesture heuristics
  ├── inference/
  │   ├── inference_image.py
  │   └── webcam_inference.py
  ├── gesture/
  │   └── classifier.py
  ├── mac_control/
  │   └── control.py
  └── README.md
  ```
- [ ] Xóa các file nháp, script test không cần thiết
- [x] Đảm bảo mọi file đều có import rõ ràng, không hardcode đường dẫn tuyệt đối

### README.md

- [x] Viết `README.md` đầy đủ:
  - [x] **Mô tả dự án:** mục tiêu, ràng buộc không dùng MediaPipe, pipeline tổng quát
  - [ ] **Demo GIF/ảnh:** dán ảnh từ Checkpoint 7.2 vào đầu README
  - [ ] **Kết quả đạt được:** cập nhật metrics thực tế sau khi train xong
  - [x] **Cài đặt môi trường:** `pip install -r requirements.txt`
  - [x] **Cách chạy training:** `python training/train_landmark.py ...`
  - [x] **Cách chạy inference:** `python inference/inference_image.py --image path/to/hand.jpg`
  - [x] **Cấu trúc thư mục:** giải thích ngắn gọn từng folder
- [ ] Commit và push lên GitHub

---

## Tham khảo: Xử lý nếu kẹt ở Checkpoint

- [ ] **Tổng nguyên tắc:** Dành tối đa 2-3 tiếng/checkpoint. Nếu quá thời gian, ghi rõ vào notes và chuyển sang bước tiếp theo.
- [ ] **Kẹt ở Phép chiếu 3D/2D:** Vẽ trục tọa độ ra giấy. Kiểm tra `K` có bị sai `fx`, `fy`, `cx`, `cy` sau khi resize ảnh không.
- [ ] **Kẹt ở Overfit test:** Giảm `batch_size=2`, tắt hết augmentation, kiểm tra scale labels (`pred` ra 0-64 nhưng `gt` lại 0-1?).
- [ ] **Gesture bị nhảy liên tục:** Tăng debounce frame và thêm cooldown theo command.
- [ ] **Bắn nhầm lệnh nhạc:** Tăng confidence threshold, thêm class `unknown`, giảm độ nhạy mapper.
