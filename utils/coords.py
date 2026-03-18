"""2.5D coordinate transformations for hand pose estimation.

This module implements Formulas 1 & 2 from the paper for converting between
3D coordinates and 2.5D representation.

2.5D Representation:
    - (u, v): 2D pixel coordinates
    - z_rel: relative depth normalized by scale factor s
    - s: scale factor = bone length from wrist (joint 0) to middle finger MCP (joint 9)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Tuple

# Joint indices according to FreiHAND/MANO topology
# Joint 0: wrist
# Joint 9: middle finger MCP (metacarpophalangeal joint)
WRIST_JOINT = 0
MIDDLE_FINGER_MCP_JOINT = 9


def compute_scale_factor(xyz: np.ndarray) -> float:
    """Compute scale factor as bone length from wrist to middle finger MCP.

    Args:
        xyz: Array of shape (21, 3) containing 3D joint positions

    Returns:
        Scale factor s (bone length in mm)
    """
    wrist = xyz[WRIST_JOINT]
    middle_mcp = xyz[MIDDLE_FINGER_MCP_JOINT]
    bone_length = np.linalg.norm(middle_mcp - wrist)
    return float(bone_length)


def xyz_to_25D(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert 3D coordinates to 2.5D representation (Formula 1).

    Args:
        xyz: Array of shape (N, 3) containing 3D joint positions in mm
             Joint 0 should be the wrist joint
        K: Camera intrinsic matrix of shape (3, 3)

    Returns:
        Array of shape (N, 3) containing [u, v, z_rel] for each joint
        - u, v: pixel coordinates
        - z_rel: relative depth normalized by scale factor s

    Formula 1:
        u = (x * fx + cx) / z
        v = (y * fy + cy) / z
        z_rel = (z - z_root) / s

    where:
        - z_root is the depth of the wrist joint (joint 0)
        - s is the scale factor (bone length wrist -> middle finger MCP)
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)

    if xyz.ndim == 1:
        xyz = xyz.reshape(1, 3)

    if xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz shape (N, 3), got {xyz.shape}")

    # Step 1: Compute wrist-relative coordinates
    xyz_root_relative = xyz - xyz[WRIST_JOINT]  # Shape: (N, 3)

    # Step 2: Compute scale factor s = bone length from wrist to middle finger MCP
    s = compute_scale_factor(xyz)

    # Step 3: Project to 2D using camera intrinsics
    # Use the same projection function as in transforms.py
    z = np.clip(xyz[:, 2:3], 1e-6, None)  # Avoid division by zero
    normalized_xyz = xyz / z
    uvw = (K @ normalized_xyz.T).T
    uv = uvw[:, :2]

    # Step 4: Compute relative depth
    z_root = xyz[WRIST_JOINT, 2]  # Depth of wrist joint
    z_rel = (xyz[:, 2] - z_root) / s

    # Combine into 2.5D representation
    uvz = np.concatenate([uv, z_rel.reshape(-1, 1)], axis=1)

    return uvz


def uvz_to_xyz(
    uvz: np.ndarray,
    K: np.ndarray,
    root_z: float,
    scale: float,
) -> np.ndarray:
    """Convert 2.5D representation back to 3D coordinates (Formula 2).

    Args:
        uvz: Array of shape (N, 3) containing [u, v, z_rel] for each joint
        K: Camera intrinsic matrix of shape (3, 3)
        root_z: Absolute depth of the root (wrist) joint in mm
        scale: Scale factor s (bone length used in xyz_to_25D)

    Returns:
        Array of shape (N, 3) containing 3D coordinates in mm

    Formula 2:
        z = z_rel * s + z_root
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
    """
    uvz = np.asarray(uvz, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)

    if uvz.ndim == 1:
        uvz = uvz.reshape(1, 3)

    if uvz.shape[1] != 3:
        raise ValueError(f"Expected uvz shape (N, 3), got {uvz.shape}")

    # Extract intrinsics
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Extract 2.5D coordinates
    u = uvz[:, 0]
    v = uvz[:, 1]
    z_rel = uvz[:, 2]

    # Step 1: Restore absolute depth
    z = z_rel * scale + root_z  # Shape: (N,)

    # Step 2: Back-project to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Combine into 3D coordinates
    xyz = np.stack([x, y, z], axis=1)

    return xyz


def compute_relative_depth(xyz: np.ndarray, root_idx: int = 0) -> np.ndarray:
    """Compute relative depth for all joints with respect to a root joint.

    Args:
        xyz: Array of shape (N, 3) containing 3D joint positions
        root_idx: Index of the root joint (default: 0 for wrist)

    Returns:
        Array of shape (N,) containing relative depth z - z_root
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    z_root = xyz[root_idx, 2]
    return xyz[:, 2] - z_root


def normalize_by_scale(xyz: np.ndarray, scale: float) -> np.ndarray:
    """Normalize 3D coordinates by scale factor.

    Args:
        xyz: Array of shape (N, 3) containing 3D joint positions
        scale: Scale factor to divide by

    Returns:
        Array of shape (N, 3) with normalized coordinates
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    return xyz / scale


# ============================================================================
# Unit Tests
# ============================================================================

def _test_xyz_to_25D_roundtrip():
    """Test that xyz -> 2.5D -> xyz roundtrip preserves coordinates."""
    # Test with synthetic hand joints (no MANO model needed)
    print("Using synthetic hand joints for testing...")

    # Create synthetic hand joints for testing
    # Simple hand-like structure
    np.random.seed(42)
    
    # Generate realistic hand positions (in mm)
    # Wrist at origin, fingers extending in +X direction
    xyz = np.zeros((21, 3))
    
    # Wrist (joint 0)
    xyz[0] = [0, 0, 500]
    
    # Finger bones (simplified)
    # Thumb
    xyz[1] = [20, -30, 500]   # CMC
    xyz[2] = [25, -35, 510]   # MCP
    xyz[3] = [30, -40, 520]   # PIP
    xyz[4] = [35, -45, 530]   # DIP
    
    # Index finger
    xyz[5] = [40, -10, 500]   # MCP
    xyz[6] = [45, -10, 515]   # PIP
    xyz[7] = [50, -10, 530]   # DIP
    xyz[8] = [55, -10, 545]   # TIP
    
    # Middle finger (joint 9 = MCP for scale calculation)
    xyz[9] = [40, 0, 500]     # MCP
    xyz[10] = [45, 0, 520]    # PIP
    xyz[11] = [50, 0, 540]    # DIP
    xyz[12] = [55, 0, 560]    # TIP
    
    # Ring finger
    xyz[13] = [40, 10, 500]   # MCP
    xyz[14] = [45, 10, 515]   # PIP
    xyz[15] = [50, 10, 530]   # DIP
    xyz[16] = [55, 10, 545]   # TIP
    
    # Pinky
    xyz[17] = [38, 20, 500]   # MCP
    xyz[18] = [42, 22, 510]   # PIP
    xyz[19] = [46, 24, 520]  # DIP
    xyz[20] = [50, 26, 530]   # TIP

    # Use a sample K matrix (typical for FreiHAND)
    K = np.array([
        [1386.854, 0, 224.409],
        [0, 1386.854, 224.409],
        [0, 0, 1]
    ], dtype=np.float32)

    print(f"Testing with xyz shape: {xyz.shape}")
    print(f"K matrix:\n{K}")

    # Test 1: xyz -> 2.5D
    uvz = xyz_to_25D(xyz, K)
    print(f"uvz shape: {uvz.shape}")
    print(f"uvz sample (first 3 joints):\n{uvz[:3]}")

    # Test 2: 2.5D -> xyz
    root_z = xyz[0, 2]  # wrist depth
    s = compute_scale_factor(xyz)
    print(f"Scale factor s: {s}")

    xyz_reconstructed = uvz_to_xyz(uvz, K, root_z, s)

    # Test 3: Compare
    error = np.max(np.abs(xyz - xyz_reconstructed))
    print(f"\nMax reconstruction error: {error}")

    if error < 1e-5:
        print("✓ TEST PASSED: Roundtrip error < 1e-5")
        return True
    else:
        print(f"✗ TEST FAILED: Roundtrip error = {error} >= 1e-5")
        print(f"Original xyz sample:\n{xyz[:3]}")
        print(f"Reconstructed xyz sample:\n{xyz_reconstructed[:3]}")
        return False


def _test_scale_factor():
    """Test scale factor computation."""
    # Create simple hand structure
    wrist = np.array([0, 0, 0])
    middle_mcp = np.array([0, 0, 100])  # 100mm bone length

    # Test with 21 joints, wrist at origin
    xyz = np.zeros((21, 3))
    xyz[0] = wrist
    xyz[9] = middle_mcp

    s = compute_scale_factor(xyz)
    expected_s = 100.0

    print(f"Scale factor: {s}, Expected: {expected_s}")
    assert abs(s - expected_s) < 1e-5, f"Scale factor mismatch: {s} != {expected_s}"
    print("✓ Scale factor test PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing 2.5D Coordinate Transformations")
    print("=" * 60)

    print("\n[Test 1] Scale Factor Computation")
    _test_scale_factor()

    print("\n[Test 2] xyz -> 2.5D -> xyz Roundtrip")
    _test_xyz_to_25D_roundtrip()

    print("\n" + "=" * 60)
    print("All tests completed")
    print("=" * 60)
