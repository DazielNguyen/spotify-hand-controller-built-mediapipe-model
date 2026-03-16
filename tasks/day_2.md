# NGÀY 2: Tích hợp MANO Parametric Hand Model

**Mục tiêu:** Nắm quyền điều khiển mô hình bàn tay 3D MANO. Đây là bước khó nhất về mặt logic của paper.

---

## Sáng: Cài đặt và Hiểu MANO

- [ ] Chọn thư viện MANO phù hợp để dùng chung với pipeline TensorFlow:
  ```bash
  pip install smplx trimesh
  ```
- [ ] Tải file model weights MANO (`MANO_RIGHT.pkl`) từ trang chủ MANO, đặt vào thư mục `models/mano/`
- [ ] Đọc và hiểu cấu trúc input của MANO:
  - [ ] `pose` (alias `theta`): tensor shape `(B, 45)` — góc xoay Axis-Angle của 15 khớp (mỗi khớp 3 chiều)
  - [ ] `betas` (alias `beta`): tensor shape `(B, 10)` — hệ số hình dạng bàn tay (dài/ngắn, mập/gầy)
  - [ ] `global_orient`: tensor shape `(B, 3)` — hướng xoay toàn bộ bàn tay trong không gian
- [ ] Viết script `scripts/test_mano.py`:
  - [ ] Khởi tạo model MANO từ thư viện đã chọn
  - [ ] Truyền vào các tensor toàn số 0: `pose=tf.zeros((1, 45))`, `betas=tf.zeros((1, 10))`
  - [ ] In shape của output: `vertices` và `joints`

> **CHECKPOINT 2.1:** Output phải trả về `vertices` shape `[1, 778, 3]` và `joints` shape `[1, 21, 3]`. Không có RuntimeError về device hay shape mismatch.

---

## Chiều: Render Mesh lên màn hình

- [ ] Cài `trimesh`: `pip install trimesh`
- [ ] Lấy tensor `vertices [1, 778, 3]` từ MANO, chuyển sang numpy
- [ ] Lấy `faces` (tam giác) từ model MANO hoặc file template mesh — shape `(1538, 3)`
- [ ] Tạo `trimesh.Trimesh(vertices=v, faces=f)` và lưu thành `outputs/hand.obj`
- [ ] Mở `hand.obj` bằng phần mềm xem 3D (Preview trên Mac / 3D Viewer trên Windows)

> **CHECKPOINT 2.2:** Nhìn thấy một bàn tay 3D tư thế thẳng (A-pose) màu xám. Nếu chỉ thấy điểm hoặc đường thẳng thì `faces` bị sai — kiểm tra lại.

---

## Tối: Đồng bộ MANO với dữ liệu FreiHAND

- [ ] Load `training_mano.json` từ FreiHAND — chứa `[theta (48,), beta (10,)]` cho từng ảnh
- [ ] Tách `global_orient = theta[:3]` và `hand_pose = theta[3:48]`
- [ ] Truyền đúng ground-truth `theta, beta` vào MANO → lấy ra `vertices [1, 778, 3]`
- [ ] Dùng hàm `project_3D_to_2D` đã viết ở Ngày 1 để chiếu 778 đỉnh mesh lên ảnh 2D
- [ ] Vẽ tất cả 778 đỉnh dạng chấm li ti (1px) màu đỏ lên ảnh RGB gốc
- [ ] Lưu ảnh ra `outputs/test_mano_overlay.jpg`

> **CHECKPOINT 2.3 (Visual Test):** Mở `test_mano_overlay.jpg`. Đám điểm đỏ li ti phải tạo thành hình bàn tay **bọc khít** toàn bộ diện tích bàn tay trong ảnh, không bị lệch ra ngoài hay thiếu vùng nào.
