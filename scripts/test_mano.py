"""Test MANO model integration.

This script verifies that the MANO parametric hand model is correctly installed
and producing the expected output shapes.

Expected output:
- vertices: shape [1, 778, 3]
- joints: shape [1, 21, 3] -> mặc định của nó là 16 nên dùng 16 trước rồi test 21 sau
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

# Fix for Python 3.11+: inspect.getargspec was removed
# chumpy library still uses it, so we add a compatibility layer
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# MANO model path
MANO_MODEL_PATH = REPO_ROOT / "models" / "mano" / "MANO_RIGHT.pkl"


def load_mano_model():
    """Load MANO model using smplx."""
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for MANO. Install with: pip install torch")

    try:
        from smplx import MANO
    except ImportError:
        raise ImportError("smplx is required. Install with: pip install smplx")

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


def test_mano_with_zeros():
    """Test MANO with zero pose and betas."""
    import torch

    print("=" * 60)
    print("Testing MANO Model Integration")
    print("=" * 60)

    # Load model
    print("\n[1] Loading MANO model...")
    mano_model = load_mano_model()
    print(f"    MANO model loaded successfully from: {MANO_MODEL_PATH}")

    # Create zero inputs (batch size = 1)
    print("\n[2] Creating zero inputs...")
    batch_size = 1

    # pose (theta): shape (B, 45) - 15 joints * 3 DOF
    pose = torch.zeros(batch_size, 45, dtype=torch.float32)
    print(f"    pose shape: {pose.shape} (should be [1, 45])")

    # betas (beta): shape (B, 10) - shape coefficients
    betas = torch.zeros(batch_size, 10, dtype=torch.float32)
    print(f"    betas shape: {betas.shape} (should be [1, 10])")

    # global_orient: shape (B, 3) - global orientation
    global_orient = torch.zeros(batch_size, 3, dtype=torch.float32)
    print(f"    global_orient shape: {global_orient.shape} (should be [1, 3])")

    # Forward pass
    print("\n[3] Running MANO forward pass...")
    with torch.no_grad():
        output = mano_model(
            betas=betas,
            hand_pose=pose,
            global_orient=global_orient,
        )

    # Extract vertices and joints
    vertices = output.vertices  # shape: [B, 778, 3]
    joints = output.joints     # shape: [B, 21, 3]

    print("\n[4] Output shapes:")
    print(f"    vertices shape: {vertices.shape} (expected: [1, 778, 3])")
    print(f"    joints shape: {joints.shape} (expected: [1, 21, 3])")

    # Verify shapes
    print("\n[5] Verification:")
    vertices_ok = vertices.shape == (1, 778, 3)
    joints_ok = joints.shape == (1, 21, 3)

    if vertices_ok and joints_ok:
        print("    ✓ CHECKPOINT 2.1 PASSED!")
        print("    ✓ vertices shape correct: [1, 778, 3]")
        print("    ✓ joints shape correct: [1, 21, 3]")
        return True
    else:
        print("    ✗ CHECKPOINT 2.1 FAILED!")
        if not vertices_ok:
            print(f"    ✗ vertices shape mismatch: got {vertices.shape}, expected [1, 778, 3]")
        if not joints_ok:
            print(f"    ✗ joints shape mismatch: got {joints.shape}, expected [1, 21, 3]")
        return False


def test_mano_with_numpy():
    """Test MANO with numpy arrays as input (alternative interface)."""
    import torch

    print("\n" + "=" * 60)
    print("Testing MANO with numpy arrays")
    print("=" * 60)

    mano_model = load_mano_model()

    # Convert numpy zeros to torch tensors
    pose_np = np.zeros((1, 45), dtype=np.float32)
    betas_np = np.zeros((1, 10), dtype=np.float32)
    global_orient_np = np.zeros((1, 3), dtype=np.float32)

    # Convert to torch
    pose = torch.from_numpy(pose_np)
    betas = torch.from_numpy(betas_np)
    global_orient = torch.from_numpy(global_orient_np)

    with torch.no_grad():
        output = mano_model(
            betas=betas,
            hand_pose=pose,
            global_orient=global_orient,
        )

    print(f"    vertices shape: {output.vertices.shape}")
    print(f"    joints shape: {output.joints.shape}")

    return True


if __name__ == "__main__":
    try:
        success = test_mano_with_zeros()
        if success:
            test_mano_with_numpy()
            print("\n" + "=" * 60)
            print("ALL TESTS PASSED!")
            print("=" * 60)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
