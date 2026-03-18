"""Loss functions for hand pose estimation training.

Implements the combined 2D + depth loss used to train the pose network:
- 2D loss: L2 between SoftArgmax-decoded heatmap coords and ground-truth (u, v)
- Depth loss: L2 between predicted relative depth and ground-truth z_rel
"""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.softargmax import SoftArgmax2D

# Re-usable layer instance
_softargmax = None


def _get_softargmax(heatmap_size: tuple[int, int] = (64, 64)) -> SoftArgmax2D:
    """Get or create a cached SoftArgmax2D instance."""
    global _softargmax
    if _softargmax is None or _softargmax.heatmap_size != heatmap_size:
        _softargmax = SoftArgmax2D(heatmap_size=heatmap_size)
    return _softargmax


def pose_loss(
    pred_heatmaps: tf.Tensor,
    pred_depth: tf.Tensor,
    gt_uvz: tf.Tensor,
    heatmap_size: int = 64,
    w_uv: float = 1.0,
    w_z: float = 1.0,
) -> tf.Tensor:
    """Compute combined 2D + depth loss for pose estimation.

    Args:
        pred_heatmaps: Predicted heatmaps, shape (B, H, W, 21)
        pred_depth: Predicted depth maps, shape (B, H, W, 21)
        gt_uvz: Ground-truth 2.5D coords in heatmap space, shape (B, 21, 3)
                 [:, :, :2] = (u, v) in [0, heatmap_size] pixel range
                 [:, :, 2]  = z_rel (relative depth)
        heatmap_size: Spatial resolution of heatmaps
        w_uv: Weight for 2D coordinate loss
        w_z: Weight for depth loss

    Returns:
        Scalar total loss
    """
    softargmax = _get_softargmax((heatmap_size, heatmap_size))

    # ── 2D Loss ────────────────────────────────────────────────────
    # SoftArgmax extracts (x, y) pixel coords from heatmaps → (B, 21, 2)
    pred_uv = softargmax(pred_heatmaps)  # (B, 21, 2)
    gt_uv = gt_uvz[:, :, :2]             # (B, 21, 2)

    loss_2d = tf.reduce_mean(tf.square(pred_uv - gt_uv))

    # ── Depth Loss ─────────────────────────────────────────────────
    # Extract per-joint depth via SoftArgmax on depth maps
    # SoftArgmax gives (B, 21, 2) spatial coords; we need weighted depth.
    # Use the softmax weights from the heatmap to read depth values.
    #
    # Strategy: flatten depth maps, apply softmax from heatmaps as weights,
    # then sum (weighted average depth per joint).
    batch_size = tf.shape(pred_heatmaps)[0]
    num_joints = pred_heatmaps.shape[-1] or 21

    # Flatten spatial dims for softmax
    hm_flat = tf.reshape(pred_heatmaps, [batch_size, -1, num_joints])  # (B, H*W, 21)
    hm_flat = tf.transpose(hm_flat, [0, 2, 1])  # (B, 21, H*W)
    hm_weights = tf.nn.softmax(hm_flat, axis=-1)  # (B, 21, H*W)

    depth_flat = tf.reshape(pred_depth, [batch_size, -1, num_joints])  # (B, H*W, 21)
    depth_flat = tf.transpose(depth_flat, [0, 2, 1])  # (B, 21, H*W)

    # Weighted sum of depths per joint → (B, 21)
    pred_z = tf.reduce_sum(hm_weights * depth_flat, axis=-1)  # (B, 21)
    gt_z = gt_uvz[:, :, 2]  # (B, 21)

    loss_depth = tf.reduce_mean(tf.square(pred_z - gt_z))

    # ── Total Loss ─────────────────────────────────────────────────
    total_loss = w_uv * loss_2d + w_z * loss_depth

    return total_loss


# ============================================================================
# Gradient Check Test (Checkpoint 4.1)
# ============================================================================

def test_gradient_check():
    """Verify that gradients flow through the loss function."""
    from models.pose_net import create_simple_pose_net

    print("=" * 60)
    print("Checkpoint 4.1: Loss Function Gradient Check")
    print("=" * 60)

    # Create a lightweight model
    model = create_simple_pose_net()

    # Dummy inputs
    batch_size = 2
    images = tf.random.normal((batch_size, 224, 224, 3))

    # Dummy ground-truth uvz in heatmap space (0-64)
    gt_uv = tf.random.uniform((batch_size, 21, 2), minval=10, maxval=54)
    gt_z = tf.random.uniform((batch_size, 21, 1), minval=-0.5, maxval=0.5)
    gt_uvz = tf.concat([gt_uv, gt_z], axis=-1)  # (B, 21, 3)

    print(f"\n[1] Input shapes:")
    print(f"    images: {images.shape}")
    print(f"    gt_uvz: {gt_uvz.shape}")

    # Forward pass + loss with gradient tape
    print("\n[2] Computing loss with GradientTape...")
    with tf.GradientTape() as tape:
        outputs = model(images, training=True)
        loss = pose_loss(outputs["heatmap"], outputs["depthmap"], gt_uvz)

    print(f"    loss value: {loss.numpy():.6f}")
    print(f"    loss shape: {loss.shape}  (should be scalar)")

    assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
    assert loss.numpy() > 0, f"Loss should be positive, got {loss.numpy()}"

    # Compute gradients
    print("\n[3] Computing gradients...")
    grads = tape.gradient(loss, model.trainable_variables)

    num_vars = len(model.trainable_variables)
    non_none_grads = sum(1 for g in grads if g is not None)

    print(f"    Trainable variables: {num_vars}")
    print(f"    Non-None gradients:  {non_none_grads}")

    assert non_none_grads > 0, "All gradients are None — no gradient flow!"

    # Check that at least some gradients have non-zero values
    non_zero_count = sum(1 for g in grads if g is not None and tf.reduce_any(g != 0).numpy())
    print(f"    Non-zero gradients:  {non_zero_count}")

    print(f"\n✓ CHECKPOINT 4.1 PASSED")
    print(f"  - Loss is positive scalar: {loss.numpy():.6f}")
    print(f"  - Gradients flow to {non_none_grads}/{num_vars} variables")


if __name__ == "__main__":
    test_gradient_check()
