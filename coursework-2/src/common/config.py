import torch
from easydict import EasyDict as edict

config = edict(
    {
        "seed": 42,  # Random seed for class-wise image selection. [42]
        "data_count": {
            "train": 15,
            "test": 15,
        },  # Data count for class-wise image selection. [15, 15]
        "device": "mps"
        if torch.backends.mps.is_available()
        else "cpu",  # Device for PyTorch tensors.
        "epoch": 50,  # Epoch for training. [50]
    }
)
