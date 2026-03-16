# NGÀY 3: Xây dựng Mạng Pose Estimation (2.5D Representation)

**Mục tiêu:** Code kiến trúc mạng (theo Bảng 5, 6, 7, 8 trong Phụ lục 7 của paper) và định dạng 2.5D (Công thức 1, 2).

---

## Sáng: Code Công Thức Tọa Độ 2.5D

- [ ] Đọc Công thức 1 & 2 trong paper, nắm rõ ký hiệu:
  - $\tilde{u}, \tilde{v}$ — tọa độ 2D trên ảnh (pixel)
  - $\tilde{z}$ — chiều sâu tương đối so với root (cổ tay), đơn vị scale bởi $s$
  - $s$ — hệ số scale, thường là chiều dài đoạn xương cổ tay → ngón giữa gốc
- [ ] Tạo file `utils/coords.py`, viết hàm `xyz_to_25D(xyz, K)`:
  - [ ] Bước 1: Tính `xyz_root_relative = xyz - xyz[0]` (trừ tọa độ cổ tay)
  - [ ] Bước 2: Tính scale $s$ = chiều dài bone tham chiếu (khớp 0 → khớp 9)
  - [ ] Bước 3: Chiếu lên 2D: `uv = project_3D_to_2D(xyz, K)` → lấy `u, v` pixel
  - [ ] Bước 4: Tính `z_rel = xyz_root_relative[:, 2] / s`
  - [ ] Output: tensor `(21, 3)` gồm `[u, v, z_rel]` cho mỗi khớp
- [ ] Viết hàm ngược `uvz_to_xyz(uvz, K, root_z, s)`:
  - [ ] Khôi phục tọa độ 3D tuyệt đối từ 2.5D
  - [ ] Dùng công thức: `x = (u - cx) * z / fx`, tương tự `y`
- [ ] Viết unit test ở cuối file:
  ```python
  xyz_orig = load_sample_xyz()  # lấy 1 sample từ dataset
  uvz = xyz_to_25D(xyz_orig, K)
  xyz_reconstructed = uvz_to_xyz(uvz, K, ...)
  assert np.max(np.abs(xyz_orig - xyz_reconstructed)) < 1e-5
  ```

> **CHECKPOINT 3.1:** Unit test pass — sai số tái tạo dưới $10^{-5}$. Nếu sai, kiểm tra lại bước chia $z$ và nhân lại `K`.

---

## Chiều: Code Network Architecture

- [ ] Tạo file `models/pose_net.py`
- [ ] Dùng ResNet18/ResNet50 pretrained làm Encoder (tải từ `torchvision.models`)
- [ ] Cắt bỏ `avgpool` và `fc` layer cuối của ResNet → giữ lại feature map
- [ ] Xây Decoder với 3 Block (theo Bảng 5-8 trong Phụ lục paper):
  - [ ] **Block0:** ConvTranspose2d upscale × 2, BatchNorm, ReLU
  - [ ] **Block1:** Skip connection từ Encoder (concatenate), ConvTranspose2d × 2
  - [ ] **Block2:** ConvTranspose2d × 2 lên `64×64`
- [ ] Chia output thành 2 nhánh song song:
  - [ ] Nhánh 1 — **2D Heatmap head**: Conv2d → shape `[B, 21, 64, 64]`
  - [ ] Nhánh 2 — **Depth map head**: Conv2d → shape `[B, 21, 64, 64]`
- [ ] Viết hàm `forward(x)` ghép Encoder → Decoder → 2 head

> **CHECKPOINT 3.2:** Chạy `model(torch.randn(2, 3, 224, 224))`. Nhận được 2 tensor output đúng shape `[2, 21, 64, 64]`. Không có RuntimeError về dimension mismatch.

---

## Tối: Viết Softargmax (Section 8.1)

- [ ] Đọc Section 8.1 — hiểu tại sao cần Softargmax thay vì `argmax` thông thường (cần differentiable)
- [ ] Tạo file `utils/softargmax.py`, viết class `SoftArgmax2D`:
  - [ ] Tạo lưới tọa độ `(x_grid, y_grid)` kích thước `H×W` với `torch.meshgrid`
  - [ ] Apply `softmax` trên heatmap (flatten spatial dims): `prob = softmax(heatmap.view(B, J, -1))`
  - [ ] Tính kỳ vọng: `x_coord = (prob * x_grid.flatten()).sum(-1)`, tương tự `y_coord`
  - [ ] Output: tensor `[B, 21, 2]` — predicted `(x, y)` pixel cho mỗi khớp
- [ ] Viết unit test:
  ```python
  heatmap = torch.zeros(1, 1, 64, 64)
  heatmap[0, 0, 20, 30] = 100.0  # peak tại (row=20, col=30)
  coords = softargmax(heatmap)    # shape [1, 1, 2]
  assert abs(coords[0,0,0] - 30.0) < 0.1  # x = col
  assert abs(coords[0,0,1] - 20.0) < 0.1  # y = row
  ```

> **CHECKPOINT 3.3:** Unit test pass. Output trả đúng `[30.0, 20.0]` (chú ý thứ tự x/col trước, y/row sau).
