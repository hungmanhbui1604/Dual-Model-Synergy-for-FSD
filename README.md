# DualModel

A dual-model fusion approach for Presentation Attack Detection (PAD) combining VGG16 and ResNet50.

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
torchrun --nproc_per_node=1 train.py --config config.yaml
```

### Key Config Options

| Section | Option | Description |
|---------|--------|-------------|
| `training` | `epochs` | Number of training epochs |
| `training` | `batch_size` | Batch size |
| `optimizer` | `lr` | Learning rate |
| `data` | `split_path` | Path to data split JSON |
| `model` | `ckpt_path` | Path to pretrained checkpoint (optional) |

## Evaluation

```bash
python evaluate.py --config config.yaml --checkpoint ckpts/best_model.pth
```

## Model Architecture

- **VGG16** (ImageNet pretrained) → 25088 features
- **ResNet50** (ImageNet pretrained) → 100352 features
- **Fusion**: Concatenation → 125440 features
- **Classifier**: 125440 → 256 → 128 → 1

## Data Format

Expected JSON split file format:
```json
{
  "train": {
    "fingerprint_id": [["image_path", label], ...],
    ...
  },
  "val": {...},
  "test": {...}
}
```

- `fingerprint_id`: e.g., `"LivDet2015_CrossMatch_41_41"`
- `label`: 0 = genuine, 1 = attack