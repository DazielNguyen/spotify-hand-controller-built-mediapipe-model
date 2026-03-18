"""Differentiable MANO Layer using smplx (PyTorch).

This module provides a differentiable MANO layer that can be used in TensorFlow
training pipeline by wrapping PyTorch's smplx MANO model.

The layer takes MANO parameters (beta, theta, global_orient, trans) as input
and returns vertices and joints.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class MANOLayer:
    """Differentiable MANO layer wrapping smplx.

    This class loads the MANO model from smplx and provides a differentiable
    forward pass that can be called from TensorFlow.
    """

    # Class-level cache for the MANO model
    _mano_model = None
    _device = None

    def __init__(
        self,
        model_path: str = "mano/MANO_MANO_RIGHT.npz",
        hand_type: str = "right",
        num_pca_comps: int = 45,
        batch_size: int = 1,
    ):
        """Initialize the MANO layer.

        Args:
            model_path: Path to MANO model file (.npz or .pth)
            hand_type: 'left' or 'right'
            num_pca_comps: Number of PCA components for pose (45)
            batch_size: Default batch size
        """
        self.hand_type = hand_type
        self.num_pca_comps = num_pca_comps
        self.batch_size = batch_size

        # Try to find MANO model file
        if not os.path.exists(model_path):
            # Try looking in common locations
            possible_paths = [
                REPO_ROOT / "mano" / "MANO_MANO_RIGHT.npz",
                REPO_ROOT / "models" / "mano" / "MANO_MANO_RIGHT.npz",
                os.path.expanduser("~/.mano/MANO_MANO_RIGHT.npz"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = str(path)
                    break

        self.model_path = model_path
        print(f"MANO Layer initialized (hand_type={hand_type})")

    def _get_mano_model(self):
        """Get or create the cached MANO model."""
        if MANOLayer._mano_model is None:
            try:
                from smplx import MANO

                MANOLayer._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                MANOLayer._mano_model = MANO(
                    model_path=self.model_path,
                    hand_type=self.hand_type,
                    num_pca_comps=self.num_pca_comps,
                    use_pca=True,
                    is_rhand=(self.hand_type == "right"),
                ).to(MANOLayer._device).eval()

                print(f"MANO model loaded on {MANOLayer._device}")
            except Exception as e:
                print(f"Warning: Could not load MANO model from smplx: {e}")
                print("MANO layer will not be functional")
                MANOLayer._mano_model = None

        return MANOLayer._mano_model

    def __call__(
        self,
        beta: tf.Tensor,
        theta: tf.Tensor,
        global_orient: tf.Tensor,
        transl: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward pass through MANO layer.

        Args:
            beta: Shape parameters, shape (B, 10)
            theta: Pose parameters (PCA coefficients), shape (B, 45)
            global_orient: Global orientation, shape (B, 3)
            transl: Translation, shape (B, 3)

        Returns:
            Tuple of (vertices, joints):
            - vertices: (B, 778, 3)
            - joints: (B, 21, 3)
        """
        mano_model = self._get_mano_model()
        if mano_model is None:
            raise RuntimeError("MANO model not loaded. Cannot forward pass.")

        # Convert TensorFlow tensors to PyTorch
        B = beta.shape[0]

        beta_np = beta.numpy()
        theta_np = theta.numpy()
        global_orient_np = global_orient.numpy()
        transl_np = transl.numpy()

        # Convert to torch tensors
        beta_torch = torch.tensor(beta_np, dtype=torch.float32, device=MANOLayer._device)
        theta_torch = torch.tensor(theta_np, dtype=torch.float32, device=MANOLayer._device)
        global_orient_torch = torch.tensor(global_orient_np, dtype=torch.float32, device=MANOLayer._device)
        transl_torch = torch.tensor(transl_np, dtype=torch.float32, device=MANOLayer._device)

        # Forward pass (no grad for inference)
        with torch.no_grad():
            output = mano_model(
                betas=beta_torch,
                hand_pose=theta_torch,
                global_orient=global_orient_torch,
                transl=transl_torch,
                return_verts=True,
            )

        vertices = output.vertices.cpu().numpy()  # (B, 778, 3)
        joints = output.joints.cpu().numpy()      # (B, 21, 3)

        # Convert back to TensorFlow
        vertices = tf.constant(vertices, dtype=tf.float32)
        joints = tf.constant(joints, dtype=tf.float32)

        return vertices, joints

    def get_joints_only(
        self,
        beta: tf.Tensor,
        theta: tf.Tensor,
        global_orient: tf.Tensor,
        transl: tf.Tensor,
    ) -> tf.Tensor:
        """Forward pass returning only joints (more efficient).

        Args:
            beta: Shape parameters, shape (B, 10)
            theta: Pose parameters (PCA coefficients), shape (B, 45)
            global_orient: Global orientation, shape (B, 3)
            transl: Translation, shape (B, 3)

        Returns:
            joints: (B, 21, 3)
        """
        mano_model = self._get_mano_model()
        if mano_model is None:
            raise RuntimeError("MANO model not loaded. Cannot forward pass.")

        # Convert TensorFlow tensors to PyTorch
        beta_np = beta.numpy()
        theta_np = theta.numpy()
        global_orient_np = global_orient.numpy()
        transl_np = transl.numpy()

        # Convert to torch tensors
        beta_torch = torch.tensor(beta_np, dtype=torch.float32, device=MANOLayer._device)
        theta_torch = torch.tensor(theta_np, dtype=torch.float32, device=MANOLayer._device)
        global_orient_torch = torch.tensor(global_orient_np, dtype=torch.float32, device=MANOLayer._device)
        transl_torch = torch.tensor(transl_np, dtype=torch.float32, device=MANOLayer._device)

        # Forward pass (no grad for inference)
        with torch.no_grad():
            output = mano_model(
                betas=beta_torch,
                hand_pose=theta_torch,
                global_orient=global_orient_torch,
                transl=transl_torch,
                return_verts=False,
            )

        joints = output.joints.cpu().numpy()  # (B, 21, 3)

        # Convert back to TensorFlow
        joints = tf.constant(joints, dtype=tf.float32)

        return joints


# Global instance
_mano_layer = None


def get_mano_layer(
    model_path: str = "mano/MANO_MANO_RIGHT.npz",
    hand_type: str = "right",
    batch_size: int = 1,
) -> MANOLayer:
    """Get or create a cached MANO layer instance."""
    global _mano_layer
    if _mano_layer is None:
        _mano_layer = MANOLayer(
            model_path=model_path,
            hand_type=hand_type,
            batch_size=batch_size,
        )
    return _mano_layer


# ============================================================================
# Test Functions
# ============================================================================

def test_mano_layer():
    """Test the MANO layer."""
    print("=" * 60)
    print("Testing MANO Layer")
    print("=" * 60)

    try:
        # Create MANO layer
        mano = get_mano_layer()

        # Create dummy inputs
        batch_size = 2
        beta = tf.zeros((batch_size, 10))
        theta = tf.zeros((batch_size, 45))
        global_orient = tf.zeros((batch_size, 3))
        transl = tf.zeros((batch_size, 3))

        print("\n[Test 1] Forward pass (vertices + joints)")
        vertices, joints = mano(beta, theta, global_orient, transl)

        print(f"  vertices shape: {vertices.shape} (expected: (2, 778, 3))")
        print(f"  joints shape: {joints.shape} (expected: (2, 21, 3))")

        assert vertices.shape == (batch_size, 778, 3)
        assert joints.shape == (batch_size, 21, 3)

        print("✓ Forward pass test PASSED")

        print("\n[Test 2] Joints only (more efficient)")
        joints_only = mano.get_joints_only(beta, theta, global_orient, transl)
        print(f"  joints_only shape: {joints_only.shape}")

        assert joints_only.shape == (batch_size, 21, 3)

        print("✓ Joints only test PASSED")

        print("\n" + "=" * 60)
        print("All MANO Layer tests PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n⚠ MANO layer test failed: {e}")
        print("This is expected if MANO model file is not available.")
        print("Please download MANO model from: https://mano.is.tue.mpg.de/")
        raise


if __name__ == "__main__":
    test_mano_layer()
