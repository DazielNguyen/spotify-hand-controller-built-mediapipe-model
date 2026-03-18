"""Softargmax implementation for differentiable heatmap decoding.

This module implements Section 8.1 from the paper - using soft-argmax instead
of argmax for differentiable coordinate extraction from heatmaps.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SoftArgmax2D(layers.Layer):
    """Differentiable 2D soft-argmax layer for heatmap decoding.

    This layer takes heatmaps of shape (B, H, W, K) where:
    - B: batch size
    - H: heatmap height
    - W: heatmap width
    - K: number of keypoints (21 for hand joints)

    And returns coordinates of shape (B, K, 2) where each coordinate
    is (x, y) in pixel space.

    The soft-argmax is computed as:
        coord = sum(softmax(heatmap) * grid)
    """

    def __init__(
        self,
        heatmap_size: Tuple[int, int] = (64, 64),
        **kwargs,
    ):
        """Initialize SoftArgmax2D layer.

        Args:
            heatmap_size: Size of input heatmaps (H, W)
        """
        super().__init__(**kwargs)
        self.heatmap_size = heatmap_size

    def build(self, input_shape):
        """Build coordinate grids."""
        height, width = self.heatmap_size

        # Create coordinate grids
        # x_grid: (1, 1, W) - column indices
        # y_grid: (1, H, 1) - row indices
        x_grid = tf.range(width, dtype=tf.float32)
        y_grid = tf.range(height, dtype=tf.float32)

        x_grid, y_grid = tf.meshgrid(x_grid, y_grid)

        # Shape: (H, W)
        self.x_grid = tf.expand_dims(x_grid, axis=0)  # (1, H, W)
        self.y_grid = tf.expand_dims(y_grid, axis=0)  # (1, H, W)

        super().build(input_shape)

    def call(self, heatmap: tf.Tensor) -> tf.Tensor:
        """Extract coordinates from heatmaps using soft-argmax.

        Args:
            heatmap: Heatmap tensor of shape (B, H, W, K)

        Returns:
            Coordinates of shape (B, K, 2) in (x, y) format
        """
        # heatmap shape: (B, H, W, K)
        batch_size = tf.shape(heatmap)[0]
        num_keypoints = heatmap.shape[-1]

        # Reshape heatmap to (B*K, H*W) for softmax
        # After softmax, reshape back to (B, K, H, W)
        heatmap_flat = tf.reshape(heatmap, [batch_size * num_keypoints, -1])  # (B*K, H*W)
        heatmap_softmax = tf.nn.softmax(heatmap_flat, axis=-1)  # (B*K, H*W)
        heatmap_softmax = tf.reshape(
            heatmap_softmax,
            [batch_size, num_keypoints, self.heatmap_size[0], self.heatmap_size[1]]
        )  # (B, K, H, W)

        # Compute expected x and y coordinates
        # x_grid, y_grid: (1, H, W) -> expand to (1, 1, H, W) for broadcasting
        x_grid_expanded = tf.expand_dims(self.x_grid, axis=0)  # (1, 1, H, W)
        y_grid_expanded = tf.expand_dims(self.y_grid, axis=0)  # (1, 1, H, W)

        # Sum over spatial dimensions (H, W)
        # heatmap_softmax: (B, K, H, W)
        # x_grid_expanded: (1, 1, H, W)
        x_coords = tf.reduce_sum(
            heatmap_softmax * x_grid_expanded,
            axis=[2, 3]
        )  # (B, K)

        y_coords = tf.reduce_sum(
            heatmap_softmax * y_grid_expanded,
            axis=[2, 3]
        )  # (B, K)

        # Stack to get (B, K, 2) with (x, y) format
        coords = tf.stack([x_coords, y_coords], axis=-1)  # (B, K, 2)

        return coords

    def get_config(self):
        """Return layer config for serialization."""
        config = super().get_config()
        config.update({
            "heatmap_size": self.heatmap_size,
        })
        return config


def softargmax_2d(
    heatmap: tf.Tensor,
    heatmap_size: Tuple[int, int] = (64, 64),
) -> tf.Tensor:
    """Functional interface for soft-argmax.

    Args:
        heatmap: Heatmap tensor of shape (B, H, W, K)
        heatmap_size: Size of heatmap (H, W)

    Returns:
        Coordinates of shape (B, K, 2) in (x, y) format
    """
    layer = SoftArgmax2D(heatmap_size=heatmap_size)
    return layer(heatmap)


def test_softargmax():
    """Test soft-argmax implementation."""
    print("=" * 60)
    print("Testing Softargmax2D")
    print("=" * 60)

    # Test parameters
    batch_size = 1
    height, width = 64, 64
    num_keypoints = 1

    # Create heatmap with peak at (row=20, col=30)
    # Note: row=y, col=x
    heatmap = tf.Variable(tf.zeros((batch_size, height, width, num_keypoints), dtype=tf.float32))

    # Set peak at (row=20, col=30)
    heatmap[0, 20, 30, 0].assign(100.0)

    # Add small values around to make softmax meaningful
    heatmap[0, 19, 30, 0].assign(50.0)
    heatmap[0, 21, 30, 0].assign(50.0)
    heatmap[0, 20, 29, 0].assign(50.0)
    heatmap[0, 20, 31, 0].assign(50.0)

    print(f"\n[Test 1] Heatmap with peak at (row=20, col=30)")
    print(f"Heatmap shape: {heatmap.shape}")

    # Apply softargmax
    layer = SoftArgmax2D(heatmap_size=(height, width))
    coords = layer(heatmap)

    print(f"Output coords shape: {coords.shape}")
    print(f"Output coords: {coords.numpy()}")

    # Check results
    x_pred = coords[0, 0, 0].numpy()
    y_pred = coords[0, 0, 1].numpy()

    print(f"\nExpected: x=30.0, y=20.0")
    print(f"Got:      x={x_pred:.2f}, y={y_pred:.2f}")

    assert abs(x_pred - 30.0) < 0.1, f"x coordinate mismatch: {x_pred} != 30.0"
    assert abs(y_pred - 20.0) < 0.1, f"y coordinate mismatch: {y_pred} != 20.0"

    print("✓ Test 1 PASSED")

    # Test with multiple keypoints
    print(f"\n[Test 2] Multiple keypoints")
    batch_size = 2
    num_keypoints = 21
    heatmap = tf.random.normal((batch_size, height, width, num_keypoints))

    coords = layer(heatmap)

    print(f"Input heatmap shape: {heatmap.shape}")
    print(f"Output coords shape: {coords.shape}")

    assert coords.shape == (batch_size, num_keypoints, 2), \
        f"Expected shape ({batch_size}, {num_keypoints}, 2), got {coords.shape}"

    print("✓ Test 2 PASSED")

    # Test with sharp peak (like in unit test requirement)
    print(f"\n[Test 3] Sharp peak test (like in requirements)")
    heatmap = tf.Variable(tf.zeros((1, 64, 64, 1), dtype=tf.float32))
    heatmap[0, 20, 30, 0].assign(100.0)  # peak at (row=20, col=30)

    coords = layer(heatmap)

    x_pred = coords[0, 0, 0].numpy()
    y_pred = coords[0, 0, 1].numpy()

    print(f"Expected: x=30.0 (col), y=20.0 (row)")
    print(f"Got:      x={x_pred:.4f}, y={y_pred:.4f}")

    assert abs(x_pred - 30.0) < 0.1, f"x mismatch: {x_pred} != 30.0"
    assert abs(y_pred - 20.0) < 0.1, f"y mismatch: {y_pred} != 20.0"

    print("✓ Test 3 PASSED")

    # Test functional API
    print(f"\n[Test 4] Functional API")
    heatmap = tf.Variable(tf.zeros((1, 64, 64, 1), dtype=tf.float32))
    heatmap[0, 20, 30, 0].assign(100.0)

    coords = softargmax_2d(heatmap, heatmap_size=(64, 64))

    print(f"Output: {coords.numpy()}")
    assert coords.shape == (1, 1, 2)
    assert abs(coords[0, 0, 0] - 30.0) < 0.1
    assert abs(coords[0, 0, 1] - 20.0) < 0.1

    print("✓ Test 4 PASSED")

    print("\n" + "=" * 60)
    print("All Softargmax tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_softargmax()
