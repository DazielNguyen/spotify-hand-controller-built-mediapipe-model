# NGÀY 1: Làm chủ Data Pipeline và Phép chiếu (Projection)

**Mục tiêu:** Hiểu hệ tọa độ 3D/2D và chuẩn bị Data Loader. Bài toán 3D rất dễ sai ở bước nhân ma trận, phải test thật kỹ.

---

## Sáng: Tải dữ liệu & Viết Script đọc JSON

- [x] Tải FreiHAND dataset về máy, giải nén vào thư mục `/dataset/freihand/`
- [x] Đọc và hiểu cấu trúc các file JSON:
  - [x] `training_K.json` — ma trận intrinsics camera (3x3) cho từng ảnh
  - [x] `training_xyz.json` — tọa độ 3D của 21 khớp tay (đơn vị: mét)
  - [x] `training_mano.json` — tham số MANO (`theta`, `beta`) ground-truth
- [x] Viết script `scripts/explore_data.py` để load và in thử:
  - [x] In tọa độ `xyz` của khớp số 0 (cổ tay) của ảnh index 0
  - [x] In ma trận `K` (3x3) của ảnh index 0
  - [x] In shape của toàn bộ tensor keypoints: phải là `(32560, 21, 3)`

> **CHECKPOINT 1.1:** Script chạy không có lỗi `FileNotFoundError` hay `KeyError`. In ra terminal thấy đủ 3 thông tin trên.

---

## Chiều: Phép chiếu 3D → 2D (Perspective Projection)

- [x] Ôn lại công thức Perspective Projection: $uv = K \cdot xyz / z$ (trong đó $z$ là chiều sâu)
- [x] Viết hàm `project_3D_to_2D(xyz: np.ndarray, K: np.ndarray) -> np.ndarray`:
  - [x] Input: `xyz` shape `(21, 3)`, `K` shape `(3, 3)`
  - [x] Chia toàn bộ `xyz` cho $z$ (tọa độ thứ 3) trước, sau đó nhân với `K`
  - [x] Output: `uv` shape `(21, 2)` — pixel coordinates trên ảnh
- [x] Dùng OpenCV (`cv2.imread`) đọc ảnh RGB tương ứng
- [x] Vẽ 21 điểm 2D lên ảnh với `cv2.circle` (màu xanh lá, bán kính 4px)
- [x] Nối các điểm thành bộ xương (skeleton) theo đúng topology 21 khớp MediaPipe/FreiHAND
- [x] Lưu ảnh kết quả ra `outputs/test_projection.jpg`

> **CHECKPOINT 1.2 (Visual Test):** Mở `test_projection.jpg`. 21 dấu chấm phải nằm **chính xác** trên các khớp ngón tay — không được lệch. Sai 1 pixel cũng phải debug lại hàm projection.

---

## Tối: Xây dựng TensorFlow `tf.data.Dataset`

- [x] Tạo file `data/dataset.py`, viết pipeline với `tf.data.Dataset`:
  - [x] Viết hàm load danh sách đường dẫn ảnh, load `K` và `xyz` từ JSON
  - [x] Tạo `tf.data.Dataset.from_tensor_slices(...)` từ paths và labels
  - [x] Viết hàm parse sample trả về `image_tensor`, `K_matrix`, `keypoints_3d`
- [x] Thêm Data Augmentation với `tf.image` hoặc Keras preprocessing layers:
  - [x] Random Crop (crop vuông bao quanh vùng tay)
  - [x] Resize về `224x224`
  - [x] Color Jitter (Brightness, Contrast, Saturation)
  - [x] **Quan trọng:** Khi crop/resize ảnh, phải cập nhật lại `fx`, `fy`, `cx`, `cy` trong ma trận `K` theo tỷ lệ crop tương ứng
- [x] Hoàn thiện pipeline dataset:
  - [x] `.shuffle(buffer_size=1000)`
  - [x] `.batch(16)`
  - [x] `.prefetch(tf.data.AUTOTUNE)`
- [x] Viết test nhanh ở cuối file:
  ```python
  batch = next(iter(dataset))
  print(batch['image'].shape)     # (16, 224, 224, 3)
  print(batch['keypoints'].shape) # (16, 21, 3)
  print(batch['K'].shape)         # (16, 3, 3)
  ```

> **CHECKPOINT 1.3:** Chạy `python data/dataset.py`. Output in đúng 3 shape ở trên, không có lỗi dimension hay key.
