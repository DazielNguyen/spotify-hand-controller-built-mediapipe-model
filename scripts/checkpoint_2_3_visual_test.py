"""MANO overlay test - Project ground-truth MANO mesh vertices onto FreiHAND images."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import DEFAULT_DATASET_ROOT
from data.transforms import project_3D_to_2D


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    dataset_root = DEFAULT_DATASET_ROOT

    # ── Load FreiHAND annotations ──────────────────────────────────
    print("Loading FreiHAND annotations...")
    intrinsics = np.asarray(load_json(dataset_root / "training_K.json"), dtype=np.float32)
    keypoints_3d = np.asarray(load_json(dataset_root / "training_xyz.json"), dtype=np.float32)

    print("Loading training_verts.json (this may take a moment)...")
    verts_all = load_json(dataset_root / "training_verts.json")
    print(f"  Loaded {len(verts_all)} samples, each with {len(verts_all[0])} vertices")

    # ── Select sample ──────────────────────────────────────────────
    idx = 5
    K = intrinsics[idx]
    xyz = keypoints_3d[idx]                          # (21, 3)
    vertices = np.asarray(verts_all[idx], dtype=np.float32)  # (778, 3)

    print(f"\n[1] Sample index: {idx}")
    print(f"    K:\n{K}")
    print(f"    xyz (21 joints) shape: {xyz.shape}")
    print(f"    vertices (778)  shape: {vertices.shape}")
    print(f"    Vertex X range: [{vertices[:, 0].min():.4f}, {vertices[:, 0].max():.4f}]")
    print(f"    Vertex Y range: [{vertices[:, 1].min():.4f}, {vertices[:, 1].max():.4f}]")
    print(f"    Vertex Z range: [{vertices[:, 2].min():.4f}, {vertices[:, 2].max():.4f}]")

    # ── Load image ─────────────────────────────────────────────────
    image_name = f"{idx:08d}.jpg"
    image_path = dataset_root / "training" / "rgb" / image_name
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    print(f"\n[2] Loaded image: {image_path.name}  shape={image.shape}")

    # ── Project to 2D ──────────────────────────────────────────────
    uv_verts = project_3D_to_2D(vertices, K)   # (778, 2)
    uv_joints = project_3D_to_2D(xyz, K)        # (21, 2)

    print(f"\n[3] Projected UV:")
    print(f"    Vertices  U: [{uv_verts[:, 0].min():.1f}, {uv_verts[:, 0].max():.1f}]  "
          f"V: [{uv_verts[:, 1].min():.1f}, {uv_verts[:, 1].max():.1f}]")
    print(f"    Joints    U: [{uv_joints[:, 0].min():.1f}, {uv_joints[:, 0].max():.1f}]  "
          f"V: [{uv_joints[:, 1].min():.1f}, {uv_joints[:, 1].max():.1f}]")

    # ── Draw overlay ───────────────────────────────────────────────
    canvas = image.copy()

    # Red dots: mesh vertices (778 points)
    for pt in uv_verts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
            cv2.circle(canvas, (x, y), radius=1, color=(0, 0, 255), thickness=-1)

    # Green dots: 21 joints
    for pt in uv_joints:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
            cv2.circle(canvas, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

    # ── Save output ────────────────────────────────────────────────
    output_path = REPO_ROOT / "outputs" / "test_mano_overlay_10.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)

    print(f"\n[4] Saved overlay to: {output_path}")
    print("\n" + "=" * 60)
    print("CHECKPOINT 2.3: Visual Test Complete")
    print("=" * 60)
    print("Open test_mano_overlay_10.jpg and verify:")
    print("- Red dots form hand mesh covering entire hand area")
    print("- Green dots are 21 joints for reference")
    print("- Mesh should align with the hand in the image")


if __name__ == "__main__":
    main()
