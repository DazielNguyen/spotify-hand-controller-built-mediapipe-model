# NGÀY 4: Training Mạng Pose & The "Overfit Test"

**Mục tiêu:** Đảm bảo Loss function đúng và mạng có khả năng học, tránh việc train mất thời gian mà model bị mù.

---

## Sáng: Định Nghĩa Loss Function

- [ ] Tạo file `training/losses.py`
- [ ] Viết hàm `pose_loss(pred_heatmaps, pred_depth, gt_uvz)`:
  - [ ] **Nhánh 2D loss:** Đưa `pred_heatmaps [B, 21, 64, 64]` qua `SoftArgmax2D` → tọa độ pixel `[B, 21, 2]`
  - [ ] Tính L2 loss giữa tọa độ 2D dự đoán vs ground-truth `uv` (từ hàm `xyz_to_25D`)
  - [ ] **Nhánh Depth loss:** Đưa `pred_depth [B, 21, 64, 64]` qua `SoftArgmax2D` theo chiều depth → `[B, 21]`
  - [ ] Tính L2 loss giữa depth dự đoán vs ground-truth `z_rel`
  - [ ] Tổng loss: `loss_total = w_uv * loss_2d + w_z * loss_depth` (bắt đầu `w_uv=1.0, w_z=1.0`)
- [ ] Kiểm tra computational graph:
  ```python
  loss = pose_loss(pred_hm, pred_d, gt_uvz)
  print(loss.shape)           # torch.Size([]) — scalar
  print(loss.requires_grad)   # True
  loss.backward()             # không có lỗi
  ```

> **CHECKPOINT 4.1:** `loss.backward()` chạy không lỗi. Loss là scalar dương. `requires_grad=True`.

---

## Chiều: Bài Test Sinh Tử — Overfit on 1 Batch

- [ ] Tạo file `training/train_pose.py`
- [ ] Tạo `DataLoader` với `batch_size=8`, `shuffle=False`
- [ ] Lấy **đúng 1 batch duy nhất**: `fixed_batch = next(iter(dataloader))`
- [ ] Viết training loop chạy 200 epochs chỉ trên `fixed_batch`:
  ```python
  for epoch in range(200):
      optimizer.zero_grad()
      pred_hm, pred_d = model(fixed_batch['image'])
      loss = pose_loss(pred_hm, pred_d, fixed_batch['uvz'])
      loss.backward()
      optimizer.step()
      writer.add_scalar('Loss/overfit', loss.item(), epoch)
  ```
- [ ] Dùng `torch.optim.Adam` với `lr=1e-3`
- [ ] Cài đặt `TensorBoard` logging: `tensorboard --logdir=runs/`
- [ ] Quan sát đồ thị Loss trên TensorBoard

> **CHECKPOINT 4.2:** Đồ thị Loss trên TensorBoard phải cắm đầu xuống gần 0 trước epoch 200. **Tuyệt đối không chuyển sang Full Training nếu chưa pass bước này** — nếu loss không giảm, nghĩa là model hoặc loss function đang có bug.

**Debug nếu Loss không giảm:**

- [ ] Kiểm tra labels: `gt_uvz` có bị normalize sai scale không? (pred ra 0-64 nhưng gt lại 0-1?)
- [ ] Kiểm tra learning rate: thử tăng lên `1e-2`
- [ ] Kiểm tra gradient: `print(loss.grad_fn)` — phải không phải `None`
- [ ] Tắt hết augmentation khi overfit test

---

## Tối: Full Training

- [ ] Bật lại full `DataLoader` với toàn bộ dataset, `shuffle=True`
- [ ] Cài Learning Rate Scheduler: `torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)`
- [ ] Cài checkpoint tự động sau mỗi epoch:
  ```python
  torch.save({'epoch': epoch, 'model': model.state_dict(),
              'optimizer': optimizer.state_dict(), 'loss': loss},
             f'checkpoints/pose_epoch_{epoch:03d}.pth')
  ```
- [ ] Chỉ giữ lại **best checkpoint** (loss thấp nhất trên validation set)
- [ ] Chạy training (để máy chạy qua đêm nếu cần)
- [ ] Theo dõi `Train Loss` và `Val Loss` trên TensorBoard đề phòng overfitting
