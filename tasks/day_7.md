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
- [ ] Tích hợp vào `FreiHANDDataset.__getitem__` với xác suất 50% (giữ green screen 50% còn lại)
- [ ] Viết script debug: vẽ 1 batch 16 ảnh ra grid, lưu `outputs/bg_augmentation_grid.jpg`

> **CHECKPOINT 7.1:** Mở `bg_augmentation_grid.jpg`. Khoảng một nửa số ảnh phải thấy bàn tay nằm trên nền tự nhiên (rừng, đường, ...). Chất lượng compositing phải tự nhiên, không thấy viền xanh hay artifact.

---

## Chiều: Real-time / Inference Demo

- [ ] Cài `pyrender` để vẽ mesh 3D: `pip install pyrender`
- [ ] Tạo file `inference/inference_image.py`:
  - [ ] Load ảnh từ path: `python inference_image.py --image my_hand.jpg`
  - [ ] Preprocess: crop vuông, resize 224×224, normalize theo ImageNet
  - [ ] Forward qua Shape Network → `beta [10]`, `theta [45]`, `global_orient [3]`, `trans [3]`
  - [ ] Truyền params vào MANO → `vertices [778, 3]`
  - [ ] Chiếu `vertices` lên 2D bằng ma trận K ước tính
  - [ ] Dùng `pyrender` hoặc `matplotlib` + `trimesh` để render mesh 3D đè lên ảnh gốc:
    ```python
    scene = pyrender.Scene()
    mesh = pyrender.Mesh.from_trimesh(hand_trimesh)
    scene.add(mesh)
    renderer = pyrender.OffscreenRenderer(224, 224)
    color, depth = renderer.render(scene)
    overlay = cv2.addWeighted(original_image, 0.7, color, 0.3, 0)
    ```
  - [ ] Lưu ảnh kết quả ra `outputs/inference_result.jpg`
- [ ] Chụp ảnh tay bạn với ít nhất 3 pose khác nhau:
  - [ ] Chữ V (Victory)
  - [ ] Nắm đấm
  - [ ] Tay xòe thẳng
- [ ] Chạy inference và lưu kết quả cho cả 3 pose

> **CHECKPOINT 7.2:** Có ít nhất 1 ảnh bản thân giơ tay chữ V với mesh 3D dự đoán đè lên tay **trông hợp lý**. Đây là bằng chứng chính cho demo project.

---

## Tối: Tổng kết Research & Dọn dẹp Code

### Cấu trúc thư mục

- [ ] Dọn dẹp và tổ chức lại cấu trúc project:
  ```
  ├── data/
  │   ├── dataset.py          # FreiHANDDataset
  │   └── transforms.py       # augmentation transforms
  ├── models/
  │   ├── pose_net.py
  │   └── shape_net.py
  ├── training/
  │   ├── losses.py
  │   ├── train_pose.py
  │   └── train_shape.py
  ├── evaluation/
  │   ├── eval_pose.py
  │   └── eval_shape.py
  ├── utils/
  │   ├── coords.py           # xyz_to_25D, projection
  │   └── softargmax.py
  ├── inference/
  │   └── inference_image.py
  └── README.md
  ```
- [ ] Xóa các file nháp, script test không cần thiết
- [ ] Đảm bảo mọi file đều có import rõ ràng, không hardcode đường dẫn tuyệt đối

### README.md

- [ ] Viết `README.md` đầy đủ:
  - [ ] **Paper được implement:** tên paper, tác giả, link arxiv
  - [ ] **Demo GIF/ảnh:** dán ảnh từ Checkpoint 7.2 vào đầu README
  - [ ] **Kết quả đạt được:** PCK@20mm = XX%, Mean Mesh Error = X.XX cm
  - [ ] **Cài đặt môi trường:** `pip install -r requirements.txt`
  - [ ] **Cách chạy training:** command line cho Pose Net và Shape Net
  - [ ] **Cách chạy inference:** `python inference/inference_image.py --image path/to/hand.jpg`
  - [ ] **Cấu trúc thư mục:** giải thích ngắn gọn từng folder
- [ ] Commit và push lên GitHub

---

## Tham khảo: Xử lý nếu kẹt ở Checkpoint

- [ ] **Tổng nguyên tắc:** Dành tối đa 2-3 tiếng/checkpoint. Nếu quá thời gian, ghi rõ vào notes và chuyển sang bước tiếp theo.
- [ ] **Kẹt ở Phép chiếu 3D/2D:** Vẽ trục tọa độ ra giấy. Kiểm tra `K` có bị sai `fx`, `fy`, `cx`, `cy` sau khi resize ảnh không.
- [ ] **Kẹt ở Overfit test:** Giảm `batch_size=2`, tắt hết augmentation, kiểm tra scale labels (`pred` ra 0-64 nhưng `gt` lại 0-1?).
- [ ] **MANO graph bị đứt:** Tìm tất cả chỗ `.detach()`, `.numpy()`, `.item()` nằm giữa pipeline, loại bỏ chúng.
- [ ] **Inference mesh lệch hoàn toàn:** Kiểm tra `global_orient` — hay bị flip trục Y/Z giữa MANO coordinate system và camera coordinate system.
