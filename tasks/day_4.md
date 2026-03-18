# NGÀY 4: Training Mạng Pose & The "Overfit Test"

**Mục tiêu:** Đảm bảo Loss function đúng và mạng có khả năng học, tránh việc train mất thời gian mà model bị mù.

---

## Sáng: Định Nghĩa Loss Function

- [x] Tạo file `training/losses.py`
- [x] Viết hàm `pose_loss(pred_heatmaps, pred_depth, gt_uvz)`:
  - [x] **Nhánh 2D loss:** Đưa `pred_heatmaps (B, 64, 64, 21)` qua `SoftArgmax2D` → tọa độ pixel `(B, 21, 2)`
  - [x] Tính L2 loss giữa tọa độ 2D dự đoán vs ground-truth `uv` (từ hàm `xyz_to_25D`)
  - [x] **Nhánh Depth loss:** Xử lý `pred_depth (B, 64, 64, 21)` để suy ra `z_rel (B, 21)`
  - [x] Tính L2 loss giữa depth dự đoán vs ground-truth `z_rel`
  - [x] Tổng loss: `loss_total = w_uv * loss_2d + w_z * loss_depth` (bắt đầu `w_uv=1.0, w_z=1.0`)
- [x] Kiểm tra computational graph:
  ```python
  with tf.GradientTape() as tape:
      loss = pose_loss(pred_hm, pred_d, gt_uvz)
  grads = tape.gradient(loss, model.trainable_variables)
  print(loss.shape)  # () — scalar
  assert all(g is not None for g in grads if g is not None)
  ```

> **CHECKPOINT 4.1:** Gradient tính được bình thường qua `tf.GradientTape`. Loss là scalar dương.

---

## Chiều: Bài Test Sinh Tử — Overfit on 1 Batch

- [x] Tạo file `training/train_pose.py`
- [x] Tạo `tf.data.Dataset` với `batch(8)`, `shuffle(False)` hoặc không shuffle
- [x] Lấy **đúng 1 batch duy nhất**: `fixed_batch = next(iter(train_dataset.take(1)))`
- [x] Viết training loop chạy 200 epochs chỉ trên `fixed_batch`:
  ```python
  for epoch in range(200):
      with tf.GradientTape() as tape:
          pred_hm, pred_d = model(fixed_batch['image'], training=True)
          loss = pose_loss(pred_hm, pred_d, fixed_batch['uvz'])
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      summary_writer.as_default()
      tf.summary.scalar('Loss/overfit', loss, step=epoch)
  ```
- [x] Dùng `tf.keras.optimizers.Adam` với `learning_rate=1e-3`
- [x] Cài đặt TensorBoard logging: `tf.summary.create_file_writer('logs/pose_overfit')`
- [x] Quan sát đồ thị Loss trên TensorBoard

> **CHECKPOINT 4.2:** Đồ thị Loss trên TensorBoard phải cắm đầu xuống gần 0 trước epoch 200. **Tuyệt đối không chuyển sang Full Training nếu chưa pass bước này** — nếu loss không giảm, nghĩa là model hoặc loss function đang có bug.

**Debug nếu Loss không giảm:**

- [x] Kiểm tra labels: `gt_uvz` có bị normalize sai scale không? (pred ra 0-64 nhưng gt lại 0-1?)
- [x] Kiểm tra learning rate: thử tăng lên `1e-2`
- [x] Kiểm tra gradient: xác nhận `tape.gradient(...)` trả ra tensor khác `None`
- [x] Tắt hết augmentation khi overfit test

---

## Tối: Full Training

- [x] Bật lại full `tf.data.Dataset` với toàn bộ dữ liệu, `shuffle(buffer_size=...)`
- [x] Cài Learning Rate Scheduler: `tf.keras.optimizers.schedules.ExponentialDecay(...)` hoặc callback `ReduceLROnPlateau`
- [x] Cài checkpoint tự động sau mỗi epoch:
  ```python
  checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
      filepath='checkpoints/pose_epoch_{epoch:03d}.weights.h5',
      save_weights_only=True,
      save_freq='epoch'
  )
  ```
- [x] Chỉ giữ lại **best checkpoint** (loss thấp nhất trên validation set)
- [x] Chạy training (để máy chạy qua đêm nếu cần)
- [x] Theo dõi `Train Loss` và `Val Loss` trên TensorBoard đề phòng overfitting
