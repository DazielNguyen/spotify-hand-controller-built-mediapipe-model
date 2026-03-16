# NGÀY 6: Đánh giá mô hình (Evaluation & Metrics)

**Mục tiêu:** Chuyển từ "code chạy được" sang "báo cáo khoa học". Phải có con số chứng minh.

---

## Sáng: Chấm điểm Pose — PCK (Percentage of Correct Keypoints)

- [ ] Tạo file `evaluation/eval_pose.py`
- [ ] Viết hàm `compute_euclidean_distance(pred_xyz, gt_xyz)`:
  - [ ] Input: `pred_xyz [N, 21, 3]`, `gt_xyz [N, 21, 3]` (đơn vị mét)
  - [ ] Output: distance array `[N, 21]` (đơn vị mét)
  - [ ] Chuyển sang mm: `dist_mm = dist * 1000`
- [ ] Viết hàm `compute_pck(pred_xyz, gt_xyz, threshold_mm=20.0)`:
  - [ ] Tính `dist_mm` cho từng cặp (pred, gt)
  - [ ] PCK = tỷ lệ keypoints có `dist_mm < threshold_mm` trên tất cả samples
  - [ ] Trả về `pck_per_joint [21]` và `pck_overall` (scalar)
- [ ] Viết script chạy eval trên toàn bộ tập test:
  - [ ] Load best checkpoint của Pose Network
  - [ ] Chạy inference không gradient (`torch.no_grad()`)
  - [ ] Tính PCK@20mm cho từng joint và overall
  - [ ] In bảng kết quả ra terminal

> **CHECKPOINT 6.1:** Script chạy xong, in ra `PCK@20mm = XX.X%`. Con số tham chiếu của paper (~85-90%). Ghi lại kết quả vào `docs/experiment_results.md`.

---

## Chiều: Chấm điểm Shape — Mesh Error & Procrustes Alignment

- [ ] Tạo file `evaluation/eval_shape.py`
- [ ] Nghiên cứu **Procrustes Alignment** — thuật toán căn chỉnh 2 point cloud:
  - [ ] Bước 1: Center về origin (trừ centroid)
  - [ ] Bước 2: Scale về unit norm
  - [ ] Bước 3: Tìm ma trận xoay tối ưu qua SVD: `U, S, Vt = np.linalg.svd(A)`
- [ ] Viết hàm `procrustes_alignment(pred_verts, gt_verts)`:
  - [ ] Input: `pred_verts [778, 3]`, `gt_verts [778, 3]`
  - [ ] Output: `pred_aligned [778, 3]` sau khi căn chỉnh
- [ ] Viết hàm `compute_mesh_error(pred_verts, gt_verts, use_procrustes=True)`:
  - [ ] Nếu `use_procrustes=True`: align trước, rồi tính khoảng cách trung bình 778 đỉnh
  - [ ] Output: `mean_error_cm` (đơn vị cm)
- [ ] Chạy eval trên tập test, in ra kết quả

> **CHECKPOINT 6.2:** In ra `Mean Mesh Error (PA) = X.XX cm`. So sánh với **1.07 cm** (MANO CNN, Bảng 3 paper). Nếu kết quả của bạn < 1.23 cm (chênh < 15%) → Tái tạo thành công!

---

## Tối: Phân tích & Gỡ lỗi

- [ ] Chạy inference trên toàn bộ tập test, lưu loss của từng sample
- [ ] Sort theo loss giảm dần → lấy top 10 ảnh tệ nhất
- [ ] Vẽ overlay mesh lên ảnh gốc cho 10 ảnh đó, lưu vào `outputs/worst_predictions/`
- [ ] Phân tích từng ảnh:
  - [ ] Tay bị che khuất (occlusion)?
  - [ ] Ánh sáng bất thường (ngược sáng, tối)?
  - [ ] Pose cực đoan (tay gập ngược, tất cả ngón duỗi thẳng)?
  - [ ] Nền bị confused với tay?
- [ ] Ghi lại nhận xét vào `docs/experiment_results.md` (section "Error Analysis")
- [ ] Đề xuất ít nhất 2 hướng cải thiện cụ thể dựa trên phân tích trên
