# NGÀY 5: Xây dựng & Train Mạng Shape Estimation

**Mục tiêu:** Mạng học cách dự đoán trực tiếp tham số MANO thay vì tọa độ (Section 5.2).

---

## Sáng: Xây dựng Shape Network

- [ ] Tạo file `models/shape_net.py`
- [ ] Load ResNet50 pretrained ImageNet:
  ```python
  backbone = torchvision.models.resnet50(pretrained=True)
  ```
- [ ] Thay thế layer `fc` cuối của ResNet50:
  - [ ] Xác định số chiều output cần: `10 (beta) + 45 (theta) + 3 (translation) + 3 (global_orient) = 61`
  - [ ] `backbone.fc = nn.Linear(backbone.fc.in_features, 61)`
- [ ] Viết hàm `forward(x)` trả về dict:
  ```python
  params = backbone(x)  # [B, 61]
  return {
      'beta':          params[:, :10],
      'theta':         params[:, 10:55],
      'global_orient': params[:, 55:58],
      'trans':         params[:, 58:61],
  }
  ```
- [ ] Viết test đường ống đầy đủ:
  - [ ] Forward 1 ảnh fake → lấy `beta, theta, global_orient`
  - [ ] Đưa vào `mano_layer` → lấy `vertices [B, 778, 3]` và `joints [B, 21, 3]`
  - [ ] Kiểm tra `joints.requires_grad == True` (graph không đứt)

> **CHECKPOINT 5.1:** Từ ảnh input, pipeline chạy liền mạch đến `joints [B, 21, 3]` với `requires_grad=True`. Không có `detach()` bị cắm vào giữa graph.

---

## Chiều: Viết Loss Function "Khủng" (Công thức 13)

- [ ] Tạo hàm `shape_loss(pred_params, gt_xyz, gt_mano_params, K)` trong `training/losses.py`:
  - [ ] **Thành phần 1 — 3D Keypoint Loss:** $L_{3D} = \|J_{pred} - J_{gt}\|^2$ (MSE trên 21 khớp 3D)
  - [ ] **Thành phần 2 — 2D Reprojection Loss:** Chiếu $J_{pred}$ lên 2D bằng K → tính MSE với `uv_gt`
  - [ ] **Thành phần 3 — Parameter Regularization Loss:** $L_p = \|\theta_{pred}\|^2 + \|\beta_{pred}\|^2$ (tránh tham số ra ngoài tầm hợp lý)
  - [ ] Tổng loss theo Công thức 13: `L = w_3d * L_3D + w_2d * L_2D + w_p * L_p`
    - `w_3d = 1000`, `w_2d = 10`, `w_p = 1`
- [ ] Kiểm tra `loss.backward()` không lỗi
- [ ] **Overfit test cho Shape Network:**
  - [ ] Cắt 1 batch cố định 8 ảnh
  - [ ] Train 200 epochs chỉ trên batch đó
  - [ ] Log loss lên TensorBoard

> **CHECKPOINT 5.2:** Loss tổng trên TensorBoard giảm liên tục trong 200 epochs. Không plateau ngay từ epoch đầu.

---

## Tối: Bắt đầu Train Mạng Shape

- [ ] Khởi tạo optimizer: `Adam` với `lr=1e-4` (nhỏ hơn Pose net do dùng pretrained weights)
- [ ] Cài `ReduceLROnPlateau` scheduler: giảm lr khi val loss không cải thiện sau 5 epoch
- [ ] Cài checkpoint save best model theo val loss
- [ ] Thêm gradient clipping để tránh exploding gradients:
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```
- [ ] Khởi chạy full training (để máy chạy qua đêm)
- [ ] Ghi lại `epoch`, `train_loss`, `val_loss` mỗi epoch vào file `logs/shape_training.csv`
