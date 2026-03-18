# NGÀY 3: Xây dựng Mạng Pose Estimation (2.5D Representation)

**Mục tiêu:** Code kiến trúc mạng (theo Bảng 5, 6, 7, 8 trong Phụ lục 7 của paper) và định dạng 2.5D (Công thức 1, 2).

---

## Sáng: Code Công Thức Tọa Độ 2.5D

- [x] Đọc Công thức 1 & 2 trong paper, nắm rõ ký hiệu:
  - $\tilde{u}, \tilde{v}$ — tọa độ 2D trên ảnh (pixel)
  - $\tilde{z}$ — chiều sâu tương đối so với root (cổ tay), đơn vị scale bởi $s$
  - $s$ — hệ số scale, thường là chiều dài đoạn xương cổ tay → ngón giữa gốc
- [x] Tạo file `utils/coords.py`, viết hàm `xyz_to_25D(xyz, K)`:
  - [x] Bước 1: Tính `xyz_root_relative = xyz - xyz[0]` (trừ tọa độ cổ tay)
  - [x] Bước 2: Tính scale $s$ = chiều dài bone tham chiếu (khớp 0 → khớp 9)
  - [x] Bước 3: Chiếu lên 2D: `uv = project_3D_to_2D(xyz, K)` → lấy `u, v` pixel
  - [x] Bước 4: Tính `z_rel = xyz_root_relative[:, 2] / s`
  - [x] Output: tensor `(21, 3)` gồm `[u, v, z_rel]` cho mỗi khớp
- [x] Viết hàm ngược `uvz_to_xyz(uvz, K, root_z, s)`:
  - [x] Khôi phục tọa độ 3D tuyệt đối từ 2.5D
  - [x] Dùng công thức: `x = (u - cx) * z / fx`, tương tự `y`
- [x] Viết unit test ở cuối file:
  ```python
  xyz_orig = load_sample_xyz()  # lấy 1 sample từ dataset
  uvz = xyz_to_25D(xyz_orig, K)
  xyz_reconstructed = uvz_to_xyz(uvz, K, ...)
  assert np.max(np.abs(xyz_orig - xyz_reconstructed)) < 1e-5
  ```

> **CHECKPOINT 3.1:** Unit test pass — sai số tái tạo dưới $10^{-5}$. Nếu sai, kiểm tra lại bước chia $z$ và nhân lại `K`.

---

## Chiều: Code Network Architecture

- [x] Tạo file `models/pose_net.py`
- [x] Dùng backbone TensorFlow/Keras pretrained làm Encoder:
  - [x] `tf.keras.applications.ResNet50` hoặc `tf.keras.applications.MobileNetV3Small`
  - [x] `include_top=False` để giữ lại feature map
- [x] Xây Decoder với 3 Block (theo Bảng 5-8 trong Phụ lục paper):
  - [x] **Block0:** `Conv2DTranspose` upscale × 2, `BatchNormalization`, `ReLU`
  - [x] **Block1:** Skip connection từ Encoder (`Concatenate`), `Conv2DTranspose` × 2
  - [x] **Block2:** `Conv2DTranspose` × 2 lên `64×64`
- [x] Chia output thành 2 nhánh song song:
  - [x] Nhánh 1 — **2D Heatmap head**: `Conv2D` → shape `(B, 64, 64, 21)`
  - [x] Nhánh 2 — **Depth map head**: `Conv2D` → shape `(B, 64, 64, 21)`
- [x] Viết hàm `call(x, training=False)` hoặc Keras Functional API ghép Encoder → Decoder → 2 head

> **CHECKPOINT 3.2:** Chạy `model(tf.random.normal((2, 224, 224, 3)))`. Nhận được 2 tensor output đúng shape `(2, 64, 64, 21)`. Không có RuntimeError về dimension mismatch.

---

## Tối: Viết Softargmax (Section 8.1)

- [x] Đọc Section 8.1 — hiểu tại sao cần Softargmax thay vì `argmax` thông thường (cần differentiable)
- [x] Tạo file `utils/softargmax.py`, viết class `SoftArgmax2D`:
  - [x] Tạo lưới tọa độ `(x_grid, y_grid)` kích thước `H×W` với `tf.meshgrid`
  - [x] Apply `tf.nn.softmax` trên heatmap sau khi reshape spatial dims
  - [x] Tính kỳ vọng: `x_coord = tf.reduce_sum(prob * x_grid_flat, axis=-1)`, tương tự `y_coord`
  - [x] Output: tensor `(B, 21, 2)` — predicted `(x, y)` pixel cho mỗi khớp
- [x] Viết unit test:
  ```python
  heatmap = tf.Variable(tf.zeros((1, 64, 64, 1), dtype=tf.float32))
  heatmap[0, 20, 30, 0].assign(100.0)  # peak tại (row=20, col=30)
  coords = softargmax(heatmap)          # shape (1, 1, 2)
  assert abs(coords[0, 0, 0] - 30.0) < 0.1  # x = col
  assert abs(coords[0, 0, 1] - 20.0) < 0.1  # y = row
  ```

> **CHECKPOINT 3.3:** Unit test pass. Output trả đúng `[30.0, 20.0]` (chú ý thứ tự x/col trước, y/row sau).
