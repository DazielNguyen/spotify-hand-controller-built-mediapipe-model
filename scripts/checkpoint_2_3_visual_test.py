"""MANO overlay test - Project MANO mesh vertices onto FreiHAND images."""

from __future__ import annotations

import sys
from pathlib import Path

# Fix numpy compatibility issue with older libraries
import numpy as np
# Patch numpy 2.x to add removed aliases
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'str'):
    np.str = str
if not hasattr(np, 'unicode'):
    np.unicode = str
if not hasattr(np, 'nan'):
    np.nan = float('nan')
if not hasattr(np, 'inf'):
    np.inf = float('inf')

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import DEFAULT_DATASET_ROOT, summarize_annotations
from data.transforms import project_3D_to_2D

# Import for compatibility with chumpy
import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec


def load_json(path: Path):
    import json
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_mano_model():
    """Load MANO model using smplx."""
    import torch
    from smplx import MANO

    MANO_MODEL_PATH = REPO_ROOT / "models" / "mano" / "MANO_RIGHT.pkl"

    if not MANO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MANO model not found at {MANO_MODEL_PATH}. "
            "Download from: https://mano.is.tue.mpg.de/"
        )

    # Load MANO model (right hand)
    mano_model = MANO(
        str(MANO_MODEL_PATH),
        model_type="mano",
        use_pca=False,
        is_rhand=True,
        num_pca_comps=45,
    )

    return mano_model


def main():
    # Load FreiHAND annotations
    dataset_root = DEFAULT_DATASET_ROOT

    print("Loading FreiHAND annotations...")
    intrinsics = np.asarray(load_json(dataset_root / "training_K.json"), dtype=np.float32)
    mano_params = np.asarray(load_json(dataset_root / "training_mano.json"), dtype=np.float32)

    print(f"MANO params shape: {mano_params.shape}")  # Should be (N, 1, 61)
    print(f"K shape: {intrinsics.shape}")  # Should be (N, 3, 3)

    # Use sample index 0 (try different indices to find a good one)
    idx = 5  # Try a different sample
    sample = mano_params[idx, 0, :]  # Shape: (61,) - take first sample and all 61 values

    # Data format: 48 (theta) + 10 (beta) + 3 (translation) = 61
    # theta: 3 (global_orient) + 45 (hand_pose)
    theta = sample[:48]  # 48 values
    beta = sample[48:58]  # 10 values
    translation = sample[58:61]  # 3 values

    # FreiHAND stores rotations as axis-angle (Rodrigues format)
    # We need to convert to rotation matrix for smplx
    import torch
    from scipy.spatial.transform import Rotation as R
    
    def axis_angle_to_rotmat(axis_angle):
        """Convert axis-angle to rotation matrix using scipy."""
        # axis_angle: (3,) or (batch, 3)
        if axis_angle.ndim == 1:
            axis_angle = axis_angle.reshape(1, 3)
        
        rot = R.from_rotvec(axis_angle)
        return rot.as_matrix()
    
    # Convert global_orient from axis-angle to rotation matrix
    global_orient_rotmat = axis_angle_to_rotmat(theta[:3])  # (1, 3, 3)
    global_orient_rotmat = torch.from_numpy(global_orient_rotmat.squeeze(0)).float()  # (3, 3)
    
    # For hand_pose (45 values = 15 joints * 3), convert each joint's axis-angle to rotmat
    hand_pose_axisangle = theta[3:48].reshape(15, 3)  # (15, 3)
    hand_pose_rotmat_list = []
    for i in range(15):
        rotmat_i = axis_angle_to_rotmat(hand_pose_axisangle[i])  # (3, 3)
        hand_pose_rotmat_list.append(torch.from_numpy(rotmat_i).float())  # (3, 3)
    hand_pose_rotmat = torch.stack(hand_pose_rotmat_list, dim=0)  # (15, 3, 3)
    
    # Use the rotation matrices
    global_orient = global_orient_rotmat
    hand_pose = hand_pose_rotmat
    
    print(f"\n[DEBUG] Using actual pose from dataset (converted to rotmat)")
    print(f"    beta: {beta}")
    print(f"    global_orient rotmat:\n{global_orient}")
    print(f"    translation: {translation}")

    # Load MANO model using smplx
    print("\n[1] Loading MANO model with smplx...")
    mano_model = load_mano_model()
    print("    ✓ MANO model loaded")

    # Prepare beta tensor (this is still numpy)
    beta_tensor = torch.from_numpy(beta.reshape(1, 10)).float()

    print(f"\n[2] Input tensors:")
    print(f"    beta: {beta_tensor.shape}")

    # Forward pass
    print("\n[3] Running MANO forward pass...")
    with torch.no_grad():
        transl_tensor = torch.from_numpy(translation.reshape(1, 3)).float()
        
        # Convert rotation matrices to 6D representation (smplx prefers this)
        # Rotation6D = concatenation of first 2 columns of rotation matrix
        def rotmat_to_6d(rotmat):
            """Convert rotation matrix to 6D representation.
            
            6D: first 2 columns of rotation matrix, flattened.
            r1, r2, r3, r4, r5, r6 (first row of first 2 cols, then second row of first 2 cols)
            """
            # rotmat: (3, 3) -> (6,)
            # First column: rotmat[:, 0] (3,)
            # Second column: rotmat[:, 1] (3,)
            return torch.cat([rotmat[:, 0], rotmat[:, 1]], dim=0)  # (6,)
        
        global_orient_aa = torch.from_numpy(theta[:3].reshape(1, 3)).float()
        
        # For hand_pose: use axis-angle directly (45 values)
        hand_pose_aa = torch.from_numpy(theta[3:48].reshape(1, 45)).float()
        
        # Translation in FreiHAND is in mm but needs scaling to meters (MANO uses meters)
        # OR it's already in a different coordinate space
        # Try scaling down translation
        translation_mm = translation  # Original in mm
        translation_scaled = translation_mm / 1000.0  # Convert to meters
        transl_tensor = torch.from_numpy(translation_scaled.reshape(1, 3)).float()
        
        print(f"    global_orient_aa shape: {global_orient_aa.shape}")
        print(f"    hand_pose_aa shape: {hand_pose_aa.shape}")
        print(f"    translation (mm): {translation_mm}")
        print(f"    translation (scaled to meters): {translation_scaled}")
        
        output = mano_model(
            betas=beta_tensor,
            hand_pose=hand_pose_aa,
            global_orient=global_orient_aa,
            transl=transl_tensor,
        )

    # Extract vertices and joints
    vertices = output.vertices.numpy()  # Shape: (1, 778, 3)
    joints = output.joints.numpy()      # Shape: (1, 16, 3) - MANO returns 16 joints

    print(f"\n[4] Output shapes:")
    print(f"    vertices: {vertices.shape} (expected: [1, 778, 3])")
    print(f"    joints: {joints.shape} (expected: [1, 16, 3])")

    # Squeeze batch dimension
    vertices = vertices.squeeze(0)  # (778, 3)
    joints = joints.squeeze(0)      # (16, 3)

    print(f"    vertices after squeeze: {vertices.shape}")
    print(f"    joints after squeeze: {joints.shape}")

    # ===== DEBUG: Check coordinate system =====
    print(f"\n[DEBUG] 3D vertices:")
    print(f"    X range: [{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]")
    print(f"    Y range: [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]")
    print(f"    Z range: [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")

    # No flip - use as-is
    # UV range [162-260, 165-248] should be in the image

    # Load image
    image_name = f"{idx:08d}.jpg"
    image_path = dataset_root / "training" / "rgb" / image_name
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    print(f"\n[5] Loaded image: {image.shape}")

    # Project vertices to 2D
    K = intrinsics[idx]
    print(f"\n[6] Intrinsics K:\n{K}")

    # Project all 778 vertices
    uv = project_3D_to_2D(vertices, K)
    print(f"    Projected UV shape: {uv.shape}")
    
    # Debug: Print vertex range and UV range
    print(f"\n[DEBUG] 3D vertices:")
    print(f"    X range: [{vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}]")
    print(f"    Y range: [{vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}]")
    print(f"    Z range: [{vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f}]")
    print(f"\n[DEBUG] Projected UV:")
    print(f"    U range: [{uv[:, 0].min():.2f}, {uv[:, 0].max():.2f}]")
    print(f"    V range: [{uv[:, 1].min():.2f}, {uv[:, 1].max():.2f}]")

    # Draw vertices on image
    canvas = image.copy()

    # Filter valid vertices (in front of camera, z > 0)
    valid_mask = vertices[:, 2] > 0
    valid_uv = uv[valid_mask]

    print(f"\n[7] Valid vertices: {valid_mask.sum()} / {len(vertices)}")

    # ===== APPLY OFFSET TO ALIGN =====
    # Use the projected UV directly (after 3D X flip)
    valid_uv_adjusted = valid_uv.copy()
    
    print(f"\n[DEBUG] After 3D X flip:")
    print(f"    U range: [{valid_uv_adjusted[:, 0].min():.2f}, {valid_uv_adjusted[:, 0].max():.2f}]")
    print(f"    V range: [{valid_uv_adjusted[:, 1].min():.2f}, {valid_uv_adjusted[:, 1].max():.2f}]")

    # Draw vertices as red dots (1px)
    for point in valid_uv_adjusted:
        x, y = int(round(point[0])), int(round(point[1]))
        if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
            cv2.circle(canvas, (x, y), radius=1, color=(0, 0, 255), thickness=-1)

    # Also draw joints (16 keypoints) in green for comparison
    uv_joints = project_3D_to_2D(joints, K)
    uv_joints_adjusted = uv_joints.copy()
    for point in uv_joints_adjusted:
        x, y = int(round(point[0])), int(round(point[1]))
        if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
            cv2.circle(canvas, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

    # Save output
    output_path = REPO_ROOT / "outputs" / "test_mano_overlay.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)

    print(f"\n[8] Saved overlay to: {output_path}")
    print("\n" + "=" * 60)
    print("CHECKPOINT 2.3: Visual Test Complete")
    print("=" * 60)
    print("Open test_mano_overlay.jpg and verify:")
    print("- Red dots form hand mesh covering entire hand area")
    print("- Green dots are 21 joints for reference")
    print("- Mesh should align with the hand in the image")


if __name__ == "__main__":
    main()
