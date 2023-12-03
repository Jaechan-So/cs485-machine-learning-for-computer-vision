import torch
from easydict import EasyDict as edict

config = edict(
    {
        "seed": 42,
        "data_count": {"train": 15, "test": 15},
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "use_pretrained": False,
        "batch_size": 128,
        "epoch": 10,
        "learning_rate": 1e-3,
        "momentum": 0.9,
        "weight_decay": 1e-5,
        "criterion": "Softmax",
    }
)
