"""Hand Pose Estimation Network with 2.5D Representation.

This module implements the network architecture based on Tables 5-8 in Appendix 7
of the paper. The network uses:
- Encoder: Pretrained backbone (ResNet50 or MobileNetV3Small)
- Decoder: 3 Upsampling blocks
- Heads: 2D Heatmap + Depth Map
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model


# Network constants
INPUT_SHAPE = (224, 224, 3)
NUM_JOINTS = 21
HEATMAP_SIZE = 64  # Output heatmap resolution


def get_encoder(
    backbone_name: str = "mobilenetv3small",
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    trainable: bool = False,
) -> Tuple[Model, int]:
    """Create encoder using pretrained backbone.

    Args:
        backbone_name: Either 'resnet50' or 'mobilenetv3small'
        input_shape: Input image shape (H, W, C)
        trainable: Whether to train the backbone

    Returns:
        Tuple of (encoder_model, encoder_output_channels)
    """
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

    backbone.trainable = trainable
    return backbone, encoder_output_channels


def decoder_block(
    x: keras.Tensor,
    skip: keras.Tensor | None,
    filters: int,
    block_name: str,
    upsample: bool = True,
) -> keras.Tensor:
    """Single decoder block with upsampling and skip connections.

    Args:
        x: Input tensor from previous layer
        skip: Skip connection tensor from encoder (or None for first block)
        filters: Number of output filters
        block_name: Name for this block
        upsample: Whether to upsample (True) or just apply conv (False)

    Returns:
        Output tensor
    """
    # Upsample if needed
    if upsample:
        x = layers.Conv2DTranspose(
            filters,
            kernel_size=4,
            strides=2,
            padding="same",
            name=f"{block_name}_upconv",
        )(x)
        x = layers.BatchNormalization(name=f"{block_name}_bn1")(x)
        x = layers.Activation("relu", name=f"{block_name}_relu1")(x)

    # Concatenate skip connection if provided
    if skip is not None:
        # Handle channel mismatch
        skip_filters = skip.shape[-1]
        if skip_filters != filters:
            skip = layers.Conv2D(
                filters,
                kernel_size=1,
                padding="same",
                name=f"{block_name}_skip_conv",
            )(skip)
        x = layers.Concatenate(name=f"{block_name}_concat")([x, skip])

    # Additional conv layers
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        name=f"{block_name}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn2")(x)
    x = layers.Activation("relu", name=f"{block_name}_relu2")(x)

    return x


def create_pose_net(
    backbone_name: str = "mobilenetv3small",
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_joints: int = NUM_JOINTS,
    heatmap_size: int = HEATMAP_SIZE,
    trainable_backbone: bool = False,
) -> Model:
    """Create full pose estimation network.

    Architecture (based on Tables 5-8):
    - Encoder: Pretrained backbone (ResNet50 or MobileNetV3Small)
    - Decoder: 3 upsampling blocks
    - Heads: 2D Heatmap (21 channels) + Depth Map (21 channels)

    Args:
        backbone_name: Encoder backbone name
        input_shape: Input image shape
        num_joints: Number of hand joints (21)
        heatmap_size: Output heatmap resolution
        trainable_backbone: Whether to train the backbone

    Returns:
        Keras model with input (224, 224, 3) and outputs:
        - heatmap: (B, 64, 64, 21)
        - depthmap: (B, 64, 64, 21)
    """
    # Encoder (pretrained backbone)
    if backbone_name == "resnet50":
        backbone = keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=None,
        )
        # Get intermediate layers for skip connections
        # Order: deepest (decoder start) -> shallower (skip connections)
        encoder_layers = [
            backbone.get_layer("conv5_block3_out").output,  # 7x7, 2048
            backbone.get_layer("conv4_block6_out").output,  # 14x14, 1024
            backbone.get_layer("conv3_block4_out").output,  # 28x28, 512
            backbone.get_layer("conv2_block3_out").output,  # 56x56, 256
        ]
    elif backbone_name == "mobilenetv3small":
        backbone = keras.applications.MobileNetV3Small(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=None,
        )
        # MobileNetV3Small layer names (Keras 3.x):
        #   activation_17       -> 7x7,   576 ch (final conv output)
        #   expanded_conv_7_add -> 14x14,  48 ch (last block before stride-2)
        #   expanded_conv_2_add -> 28x28,  24 ch (last block before stride-2)
        #   activation          -> 112x112, 16 ch (first conv)
        encoder_layers = [
            backbone.get_layer("activation_17").output,       # 7x7, 576
            backbone.get_layer("expanded_conv_7_add").output, # 14x14, 48
            backbone.get_layer("expanded_conv_2_add").output, # 28x28, 24
            backbone.get_layer("activation").output,          # 112x112, 16
        ]
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    backbone.trainable = trainable_backbone

    # Use the deepest layer as decoder input
    # encoder_layers: [0]=7x7, [1]=14x14, [2]=28x28, [3]=56/112
    x = encoder_layers[0]  # 7x7 with highest channels

    # ===== DECODER =====
    # decoder_block upsamples FIRST, then concatenates skip.
    # So skip must match the POST-upsample resolution.

    # Block 0: 7x7 → 14x14, skip from encoder at 14x14
    x = decoder_block(
        x,
        skip=encoder_layers[1],
        filters=256,
        block_name="decoder_block0",
        upsample=True,
    )  # 14x14

    # Block 1: 14x14 → 28x28, skip from encoder at 28x28
    x = decoder_block(
        x,
        skip=encoder_layers[2],
        filters=128,
        block_name="decoder_block1",
        upsample=True,
    )  # 28x28

    # Block 2: 28x28 → 56x56, no skip
    x = decoder_block(
        x,
        skip=None,
        filters=64,
        block_name="decoder_block2",
        upsample=True,
    )  # 56x56

    # Resize from 56x56 to 64x64 (target heatmap resolution)
    x = layers.Resizing(heatmap_size, heatmap_size, name="resize_to_heatmap")(x)
    x = layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        name="final_conv",
    )(x)
    x = layers.BatchNormalization(name="final_bn")(x)
    x = layers.Activation("relu", name="final_relu")(x)  # 64x64

    # ===== HEADS =====
    # 2D Heatmap head — raw logits for SoftArgmax
    heatmap = layers.Conv2D(
        num_joints,
        kernel_size=1,
        padding="same",
        name="heatmap_conv",
    )(x)

    # Depth map head — raw values (z_rel can be negative)
    depthmap = layers.Conv2D(
        num_joints,
        kernel_size=1,
        padding="same",
        name="depthmap_conv",
    )(x)

    # Create model
    model = Model(
        inputs=backbone.input,
        outputs={"heatmap": heatmap, "depthmap": depthmap},
        name="hand_pose_net",
    )

    return model


def create_simple_pose_net(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_joints: int = NUM_JOINTS,
) -> Model:
    """Create a simple pose network without pretrained backbone for testing.

    This is a lightweight version for quick testing without downloading
    pretrained weights.

    Args:
        input_shape: Input image shape
        num_joints: Number of hand joints

    Returns:
        Keras model
    """
    inputs = keras.Input(shape=input_shape, name="image")

    # Simple encoder: 224 → 112 → 56 → 28 → 14 → 7
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)

    # Decoder: 7 → 14 → 28 → 56, then resize to 64
    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Resizing(HEATMAP_SIZE, HEATMAP_SIZE)(x)

    # Heads
    heatmap = layers.Conv2D(num_joints, 1, padding="same", activation="sigmoid", name="heatmap")(x)
    depthmap = layers.Conv2D(num_joints, 1, padding="same", activation="sigmoid", name="depthmap")(x)

    model = Model(inputs=inputs, outputs={"heatmap": heatmap, "depthmap": depthmap})

    return model


# ============================================================================
# Unit Tests
# ============================================================================

def test_pose_net():
    """Test that pose network produces correct output shapes."""
    print("=" * 60)
    print("Testing Pose Network")
    print("=" * 60)

    # Test with MobileNetV3Small backbone
    print("\n[Test 1] MobileNetV3Small backbone")
    model = create_pose_net(backbone_name="mobilenetv3small")
    model.summary()

    # Test forward pass
    test_input = tf.random.normal((2, 224, 224, 3))
    outputs = model(test_input, training=False)

    print(f"\nInput shape: {test_input.shape}")
    print(f"Heatmap output shape: {outputs['heatmap'].shape}")
    print(f"Depthmap output shape: {outputs['depthmap'].shape}")

    # Verify shapes
    assert outputs["heatmap"].shape == (2, 64, 64, 21), \
        f"Expected heatmap shape (2, 64, 64, 21), got {outputs['heatmap'].shape}"
    assert outputs["depthmap"].shape == (2, 64, 64, 21), \
        f"Expected depthmap shape (2, 64, 64, 21), got {outputs['depthmap'].shape}"

    print("✓ MobileNetV3Small backbone test PASSED")

    # Test with ResNet50 backbone
    print("\n[Test 2] ResNet50 backbone")
    model = create_pose_net(backbone_name="resnet50")
    
    test_input = tf.random.normal((2, 224, 224, 3))
    outputs = model(test_input, training=False)

    print(f"Input shape: {test_input.shape}")
    print(f"Heatmap output shape: {outputs['heatmap'].shape}")
    print(f"Depthmap output shape: {outputs['depthmap'].shape}")

    assert outputs["heatmap"].shape == (2, 64, 64, 21)
    assert outputs["depthmap"].shape == (2, 64, 64, 21)

    print("✓ ResNet50 backbone test PASSED")

    # Test simple network (no pretrained weights)
    print("\n[Test 3] Simple network (no pretrained weights)")
    model = create_simple_pose_net()

    test_input = tf.random.normal((2, 224, 224, 3))
    outputs = model(test_input, training=False)

    print(f"Input shape: {test_input.shape}")
    print(f"Heatmap output shape: {outputs['heatmap'].shape}")
    print(f"Depthmap output shape: {outputs['depthmap'].shape}")

    assert outputs["heatmap"].shape == (2, 64, 64, 21)
    assert outputs["depthmap"].shape == (2, 64, 64, 21)

    print("✓ Simple network test PASSED")

    print("\n" + "=" * 60)
    print("All pose network tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_pose_net()
