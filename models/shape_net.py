"""Shape Estimation Network - MANO Parameter Prediction.

This module implements the Shape Network from Section 5.2 of the paper.
The network predicts MANO parameters (beta, theta, global_orient, translation)
directly from input images, which can be used to generate 3D hand mesh and joints.

Architecture:
- Encoder: Pretrained ResNet50 backbone (ImageNet weights)
- Head: Dense layer predicting 61 parameters:
  - beta (shape): 10 dimensions
  - theta (pose): 45 dimensions (15 joints x 3 rotations)
  - global_orient: 3 dimensions
  - translation: 3 dimensions
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


# Network constants
INPUT_SHAPE = (224, 224, 3)

# MANO parameter dimensions
BETA_DIM = 10      # Shape parameters
THETA_DIM = 45     # Pose parameters (15 joints x 3)
GLOBAL_ORIENT_DIM = 3  # Global rotation
TRANSLATION_DIM =  3   # Translation
TOTAL_PARAMS = BETA_DIM + THETA_DIM + GLOBAL_ORIENT_DIM + TRANSLATION_DIM  # 61


def create_shape_net(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    backbone_name: str = "resnet50",
    trainable_backbone: bool = True,
) -> Model:
    """Create Shape Network for MANO parameter prediction.

    Args:
        input_shape: Input image shape (224, 224, 3)
        backbone_name: Either 'resnet50' or 'mobilenetv3small'
        trainable_backbone: Whether to train the backbone

    Returns:
        Keras model with:
        - Input: (B, 224, 224, 3)
        - Output: dict with 'params' (B, 61) containing [beta, theta, global_orient, trans]
    """
    # Input
    inputs = keras.Input(shape=input_shape, name="image")

    # Encoder: Pretrained backbone
    if backbone_name == "resnet50":
        backbone = keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
        encoder_output_channels = 2048
    elif backbone_name == "mobilenetv3small":
        backbone = keras.applications.MobileNetV3Small(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
        encoder_output_channels = 576
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    backbone.trainable = trainable_backbone

    # Get backbone features
    x = backbone(inputs, training=trainable_backbone)

    # Shape/pose prediction head
    x = layers.Dense(
        256,
        activation="relu",
        name="shape_dense1",
    )(x)
    x = layers.Dropout(0.2, name="shape_dropout1")(x)

    x = layers.Dense(
        128,
        activation="relu",
        name="shape_dense2",
    )(x)
    x = layers.Dropout(0.2, name="shape_dropout2")(x)

    # Output: 61 MANO parameters
    # [beta(10) + theta(45) + global_orient(3) + trans(3) = 61]
    params = layers.Dense(
        TOTAL_PARAMS,
        name="mano_params",
    )(x)

    model = Model(inputs=inputs, outputs=params, name="shape_net")

    return model


def split_mano_params(params: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Split concatenated MANO parameters into components.

    Args:
        params: Tensor of shape (B, 61) containing [beta, theta, global_orient, trans]

    Returns:
        Tuple of (beta, theta, global_orient, trans) each of shape (B, dim)
    """
    beta = params[:, :BETA_DIM]                    # (B, 10)
    theta = params[:, BETA_DIM:BETA_DIM + THETA_DIM]  # (B, 45)
    global_orient = params[:, BETA_DIM + THETA_DIM:BETA_DIM + THETA_DIM + GLOBAL_ORIENT_DIM]  # (B, 3)
    trans = params[:, BETA_DIM + THETA_DIM + GLOBAL_ORIENT_DIM:]  # (B, 3)

    return beta, theta, global_orient, trans


# ============================================================================
# Unit Tests
# ============================================================================

def test_shape_net():
    """Test that Shape Network produces correct output shapes."""
    print("=" * 60)
    print("Testing Shape Network")
    print("=" * 60)

    # Test with ResNet50 backbone
    print("\n[Test 1] ResNet50 backbone")
    model = create_shape_net(backbone_name="resnet50", trainable_backbone=True)
    model.summary()

    # Test forward pass
    test_input = tf.random.normal((2, 224, 224, 3))
    outputs = model(test_input, training=False)

    print(f"\nInput shape: {test_input.shape}")
    print(f"Output params shape: {outputs.shape}")

    # Verify shapes
    assert outputs.shape == (2, 61), \
        f"Expected output shape (2, 61), got {outputs.shape}"

    print("✓ ResNet50 backbone test PASSED")

    # Test parameter splitting
    print("\n[Test 2] Parameter splitting")
    beta, theta, global_orient, trans = split_mano_params(outputs)
    print(f"  beta shape: {beta.shape} (expected: (2, 10))")
    print(f"  theta shape: {theta.shape} (expected: (2, 45))")
    print(f"  global_orient shape: {global_orient.shape} (expected: (2, 3))")
    print(f"  trans shape: {trans.shape} (expected: (2, 3))")

    assert beta.shape == (2, 10)
    assert theta.shape == (2, 45)
    assert global_orient.shape == (2, 3)
    assert trans.shape == (2, 3)

    print("✓ Parameter splitting test PASSED")

    # Test with MobileNetV3Small backbone
    print("\n[Test 3] MobileNetV3Small backbone")
    model = create_shape_net(backbone_name="mobilenetv3small", trainable_backbone=True)

    test_input = tf.random.normal((2, 224, 224, 3))
    outputs = model(test_input, training=False)

    print(f"Input shape: {test_input.shape}")
    print(f"Output params shape: {outputs.shape}")

    assert outputs.shape == (2, 61)

    print("✓ MobileNetV3Small backbone test PASSED")

    print("\n" + "=" * 60)
    print("All Shape Network tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_shape_net()
