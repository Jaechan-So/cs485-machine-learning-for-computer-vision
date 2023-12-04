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
        "use_pretrained": False,  # Whether use pretrained weights or not. [False]
        "batch_size": 10,  # Batch size for Dataloader. [10]
        "epoch": 50,  # Epoch for training. [50]
        "learning_rate": 1e-3,  # Learning rate for optimizer. [1e-3]
        "momentum": 0.9,  # Momentum for optimizer. [0.9]
        "weight_decay": 1e-5,  # Weight decay for L2 regularization term in optimizer. [1e-5]
        "model": "AlexNet",  # Model selection. Possible models: AlexNet, ResNet50, ResNet18
        "loss": "CrossEntropy",  # Loss selection. Possible losses: CrossEntropy, SquaredHinge
        "compress": False,  # Whether compress the fully connected layer by truncated SVD or not. [False]
    }
)
