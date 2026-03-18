"""Training script for hand pose estimation network.

Supports two modes:
  --mode overfit   : Overfit on 1 batch (Checkpoint 4.2)
  --mode full      : Full training with LR scheduler + checkpoints
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import build_dataset
from models.pose_net import create_pose_net, create_simple_pose_net, HEATMAP_SIZE
from training.losses import pose_loss
from utils.coords import xyz_to_25D


# ─── Helpers ───────────────────────────────────────────────────────

def _xyz_to_uvz_heatmap_space(xyz: np.ndarray, K: np.ndarray, image_size: int = 224) -> np.ndarray:
    """Convert 3D keypoints to 2.5D in HEATMAP space (0–64).

    Steps:
      1. xyz_to_25D → (u, v) in image-pixel space + z_rel
      2. Scale (u, v) from image space to heatmap space
    """
    uvz = xyz_to_25D(xyz, K)  # (21, 3), uv in [0, image_size]
    scale = HEATMAP_SIZE / image_size
    uvz[:, 0] *= scale  # u
    uvz[:, 1] *= scale  # v
    # z_rel stays unchanged
    return uvz.astype(np.float32)


def _prepare_batch_uvz(batch: dict[str, tf.Tensor]) -> tf.Tensor:
    """Convert a batch's xyz keypoints + K into 2.5D heatmap-space labels.

    Args:
        batch: dict with 'keypoints' (B, 21, 3) and 'K' (B, 3, 3)

    Returns:
        gt_uvz: (B, 21, 3) in heatmap pixel space
    """
    xyz_batch = batch["keypoints"].numpy()  # (B, 21, 3)
    K_batch = batch["K"].numpy()            # (B, 3, 3)
    batch_size = xyz_batch.shape[0]

    uvz_list = []
    for i in range(batch_size):
        uvz_i = _xyz_to_uvz_heatmap_space(xyz_batch[i], K_batch[i])
        uvz_list.append(uvz_i)

    return tf.constant(np.stack(uvz_list, axis=0), dtype=tf.float32)  # (B, 21, 3)


# ─── Overfit Test (Checkpoint 4.2) ─────────────────────────────────

def run_overfit_test(
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    batch_size: int = 8,
    backbone: str = "mobilenetv3small",
):
    """Overfit on a single batch — loss must drop near zero."""
    print("=" * 60)
    print("Checkpoint 4.2: Overfit on 1 Batch")
    print("=" * 60)

    # Build dataset (no shuffle, no augmentation)
    print("\n[1] Loading dataset...")
    dataset = build_dataset(
        batch_size=batch_size,
        shuffle=False,
        training=False,  # no augmentation
        limit=batch_size,  # only load 1 batch worth of data
    )

    # Get exactly 1 batch
    fixed_batch = next(iter(dataset.take(1)))
    print(f"    image shape:     {fixed_batch['image'].shape}")
    print(f"    keypoints shape: {fixed_batch['keypoints'].shape}")
    print(f"    K shape:         {fixed_batch['K'].shape}")

    # Pre-compute ground-truth uvz in heatmap space
    gt_uvz = _prepare_batch_uvz(fixed_batch)
    print(f"    gt_uvz shape:    {gt_uvz.shape}")
    print(f"    gt_uvz uv range: [{gt_uvz[:,:,:2].numpy().min():.1f}, {gt_uvz[:,:,:2].numpy().max():.1f}]")
    print(f"    gt_uvz z range:  [{gt_uvz[:,:,2].numpy().min():.3f}, {gt_uvz[:,:,2].numpy().max():.3f}]")

    # Build model
    print(f"\n[2] Creating model (backbone={backbone})...")
    model = create_pose_net(backbone_name=backbone, trainable_backbone=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    print(f"    Trainable params: {sum(v.numpy().size for v in model.trainable_variables):,}")

    # TensorBoard logging
    log_dir = REPO_ROOT / "logs" / "pose_overfit" / datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(str(log_dir))
    print(f"    TensorBoard logs: {log_dir}")

    # Training loop
    images = fixed_batch["image"]
    print(f"\n[3] Training for {num_epochs} epochs on 1 batch...")

    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            outputs = model(images, training=True)
            loss = pose_loss(outputs["heatmap"], outputs["depthmap"], gt_uvz)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar("Loss/overfit", loss, step=epoch)

        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"    Epoch {epoch:4d}/{num_epochs}  loss={loss.numpy():.6f}")

    # Final check
    final_loss = loss.numpy()
    print(f"\n[4] Final loss: {final_loss:.6f}")

    if final_loss < 1.0:
        print("✓ CHECKPOINT 4.2 PASSED — loss dropped near zero")
    else:
        print(f"⚠ CHECKPOINT 4.2 — loss={final_loss:.4f}, may need more epochs or tuning")
        print("  Debug tips:")
        print("  - Try --lr 1e-2")
        print("  - Try --epochs 500")
        print("  - Check gt_uvz scale matches heatmap space (0-64)")

    print(f"\n    Run TensorBoard: tensorboard --logdir {REPO_ROOT / 'logs'}")


# ─── Full Training ─────────────────────────────────────────────────

def run_full_training(
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    batch_size: int = 16,
    backbone: str = "mobilenetv3small",
):
    """Full training with LR scheduler and checkpointing."""
    print("=" * 60)
    print("Full Training")
    print("=" * 60)

    # Build train dataset
    print("\n[1] Loading dataset...")
    train_dataset = build_dataset(
        batch_size=batch_size,
        shuffle=True,
        training=True,
    )

    # Build model
    print(f"\n[2] Creating model (backbone={backbone})...")
    model = create_pose_net(backbone_name=backbone, trainable_backbone=True)

    # LR scheduler — exponential decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Checkpoints
    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    log_dir = REPO_ROOT / "logs" / "pose_full" / datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(str(log_dir))
    print(f"    TensorBoard logs: {log_dir}")

    best_loss = float("inf")
    global_step = 0

    for epoch in range(num_epochs):
        epoch_losses = []

        for batch in train_dataset:
            gt_uvz = _prepare_batch_uvz(batch)

            with tf.GradientTape() as tape:
                outputs = model(batch["image"], training=True)
                loss = pose_loss(outputs["heatmap"], outputs["depthmap"], gt_uvz)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_losses.append(loss.numpy())
            global_step += 1

            with summary_writer.as_default():
                tf.summary.scalar("Loss/train_step", loss, step=global_step)

        avg_loss = np.mean(epoch_losses)

        with summary_writer.as_default():
            tf.summary.scalar("Loss/train_epoch", avg_loss, step=epoch)

        print(f"  Epoch {epoch+1:3d}/{num_epochs}  avg_loss={avg_loss:.4f}  "
              f"lr={optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else learning_rate:.6f}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = ckpt_dir / "pose_best.weights.h5"
            model.save_weights(str(save_path))
            print(f"    ✓ Saved best checkpoint (loss={best_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = ckpt_dir / f"pose_epoch_{epoch+1:03d}.weights.h5"
            model.save_weights(str(save_path))

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Best weights saved at: {ckpt_dir / 'pose_best.weights.h5'}")
    print(f"TensorBoard: tensorboard --logdir {REPO_ROOT / 'logs'}")


# ─── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["overfit", "full"], default="overfit",
                        help="Training mode")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (default: 200 for overfit, 50 for full)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--backbone", default="mobilenetv3small",
                        choices=["mobilenetv3small", "resnet50"],
                        help="Encoder backbone")
    args = parser.parse_args()

    if args.mode == "overfit":
        epochs = args.epochs or 200
        run_overfit_test(
            num_epochs=epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            backbone=args.backbone,
        )
    else:
        epochs = args.epochs or 50
        run_full_training(
            num_epochs=epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            backbone=args.backbone,
        )


if __name__ == "__main__":
    main()
