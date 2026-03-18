# Walkthrough: Day 4 — Training Pipeline

## Files Created

| File | Description |
|---|---|
| [losses.py](file:///Users/vananhduy/Documents/Repository_Git_Hub/spotify-hand-controller-built-mediapipe-model/training/losses.py) | [pose_loss()](file:///Users/vananhduy/Documents/Repository_Git_Hub/spotify-hand-controller-built-mediapipe-model/training/losses.py#33-93) — SoftArgmax + L2 for 2D and depth |
| [train_pose.py](file:///Users/vananhduy/Documents/Repository_Git_Hub/spotify-hand-controller-built-mediapipe-model/training/train_pose.py) | Overfit test + full training with LR scheduler |

## Checkpoint Results

| Checkpoint | Status | Key Metric |
|---|---|---|
| 4.1 — Gradient check | ✅ Pass | Loss=195.8 (scalar), 20/20 gradients |
| 4.2 — Overfit test | ✅ Pass | Loss: 133 → 0.679 in 200 epochs |

## Critical Fix: Remove Sigmoid from Heads

The initial overfit test with sigmoid activations on heatmap/depth heads plateaued at loss ~47.

**Root cause**: [SoftArgmax2D](file:///Users/vananhduy/Documents/Repository_Git_Hub/spotify-hand-controller-built-mediapipe-model/utils/softargmax.py#16-115) needs raw logits (unbounded) to form sharp spatial peaks. Sigmoid compresses output to [0,1], limiting expressiveness. Also, depth ground truth `z_rel ∈ [-1.17, 1.47]` can't be represented by sigmoid [0,1].

**Fix**: Removed sigmoid from both heads in [pose_net.py](file:///Users/vananhduy/Documents/Repository_Git_Hub/spotify-hand-controller-built-mediapipe-model/models/pose_net.py).

| Run | Sigmoid | Epoch 60 | Final (epoch 200) |
|---|---|---|---|
| 1st | ✅ On | 49.9 | 48.6 (plateau) |
| 2nd | ❌ Off | 1.7 | 0.679 ✓ |

## Usage

```bash
# Overfit test
python training/train_pose.py --mode overfit --epochs 200

# Full training
python training/train_pose.py --mode full --epochs 50

# TensorBoard
tensorboard --logdir logs/
```
