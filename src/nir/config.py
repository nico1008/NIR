"""The fixed architecture and training protocol."""

CONFIG = dict(
    seed=42,
    size=224,
    batch_size=32,
    epochs=20,
    warmup_epochs=3,
    patience=5,
    head_warmup_lr=1e-3,
    head_lr=1e-4,
    backbone_lr=1e-5,
    weight_decay=1e-2,
    weights="densenet121-res224-all",
)
CLASSES = ["NORMAL", "PNEUMONIA"]
