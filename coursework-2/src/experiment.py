import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from easydict import EasyDict as edict
from torch import nn, optim, onnx
from torchvision.models import (
    alexnet,
    AlexNet_Weights,
    resnet50,
    ResNet50_Weights,
    resnet18,
    ResNet18_Weights,
)

from common.config import config
from model import LeNet
from preprocess import preprocess


def get_default_args():
    return edict(
        {
            "batch_size": 10,  # Batch size for Dataloader. [10]
            "use_pretrained": False,  # Whether use pretrained weights or not. [False]
            "learning_rate": 1e-3,  # Learning rate for optimizer. [1e-3]
            "momentum": 0.9,  # Momentum for optimizer. [0.9]
            "weight_decay": 1e-5,  # Weight decay for L2 regularization term in optimizer. [1e-5]
            "model": "LeNet",  # Model selection. [AlexNet, ResNet50, ResNet18, LeNet]
            "loss": "CrossEntropy",  # Loss selection. [CrossEntropy, SquaredHinge]
            "conv_channels": [6, 16],  # Channel values for convolution layers. [6, 16]
            "fc_sizes": [
                120,
                84,
                10,
            ],  # Size values for fully connected layers. [120, 84, 10]
            "kernel_size": 3,  # Kernel size for convolution layer and max pooling layer. [3]
            "skip": False,  # Whether use skip connection. [False]
            "dropout": False,  # Whether use dropout. [False]
            "norm": None,  # What normalization technique to use. [None, BatchNorm, InstanceNorm, LayerNorm]
        }
    )


def plot_image(image):
    image = (image * 0.5) + 0.5
    plt.imshow(np.array(image).transpose(1, 2, 0))
    plt.show()


def get_criterion(args):
    if args.loss == "CrossEntropy":
        return nn.CrossEntropyLoss()
    elif args.loss == "SquaredHinge":
        # Currently, PyTorch does not support multi margin loss in mps device.
        # Therefore PYTORCH_ENABLE_MPS_FALLBACK=1 environment variable is needed.
        return nn.MultiMarginLoss(p=2)

    raise Exception("No criterion selected")


def get_model(args):
    if args.model == "AlexNet":
        weights = AlexNet_Weights.DEFAULT if args.use_pretrained else None
        model = alexnet(weights=weights)
        model.classifier[6] = nn.Linear(
            in_features=4096, out_features=args.num_of_classes
        )
        return model
    elif args.model == "ResNet50":
        weights = ResNet50_Weights.DEFAULT if args.use_pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=args.num_of_classes)
        return model
    elif args.model == "ResNet18":
        weights = ResNet18_Weights.DEFAULT if args.use_pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=args.num_of_classes)
        return model
    elif args.model == "LeNet":
        return LeNet(
            conv_channels=args.conv_channels,
            fc_sizes=args.fc_sizes,
            kernel_size=args.kernel_size,
            skip=args.skip,
            dropout=args.dropout,
            norm=args.norm,
        )

    raise Exception("No model selected")


def get_context(args):
    model = get_model(args)
    criterion = get_criterion(args)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    return model, criterion, optimizer


def compress_model(model):
    for i in range(len(model.fc_blocks)):
        fc_block = model.fc_blocks[i]
        weight = fc_block.fc.weight.cpu().detach().numpy()
        bias = fc_block.fc.bias.cpu().detach().numpy()

        U, S, Vh = scipy.sparse.linalg.svds(weight, k=8)

        compressed_first_weight = torch.Tensor(np.diag(S) @ Vh).to(config.device)
        compressed_second_weight = torch.Tensor(U).to(config.device)
        compressed_bias = torch.Tensor(bias).to(config.device)

        compressed_first_layer = nn.Linear(
            in_features=compressed_first_weight.size(0),
            out_features=compressed_second_weight.size(1),
            bias=False,
        )
        compressed_first_layer.weight.data = compressed_first_weight

        compressed_second_layer = nn.Linear(
            in_features=compressed_second_weight.size(0),
            out_features=compressed_second_weight.size(1),
            bias=True,
        )
        compressed_second_layer.weight.data = compressed_second_weight
        compressed_second_layer.bias.data = compressed_bias

        model.fc_blocks[i].fc = nn.Sequential(
            compressed_first_layer, compressed_second_layer
        )


def get_parameters_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, train_loader, criterion, optimizer):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(config.device)
        labels = labels.to(config.device)

        logits = model(images)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, test_loader):
    total_accurate_cases = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(config.device)
            labels = labels.to(config.device)

            logits = model(images)
            total_accurate_cases += (logits.argmax(1) == labels).cpu().numpy().sum()

    total_items = test_loader.batch_size * len(test_loader)
    accuracy = (total_accurate_cases / total_items) * 100

    return accuracy


def train_model(
    train_loader,
    test_loader,
    model,
    criterion,
    optimizer,
):
    model = model.to(config.device)

    accuracies = []
    for epoch in range(config.epoch):
        train(model, train_loader, criterion, optimizer)
        accuracy = evaluate(model, test_loader)
        accuracies.append(accuracy)
        print(f"Epoch {epoch}, Accuracy {accuracy:.2f}%")

    return list(range(config.epoch)), accuracies, model


def run_experiment(args, title):
    (
        train_loader,
        test_loader,
        num_of_classes,
        _,
        _,
    ) = preprocess(args)
    args.num_of_classes = num_of_classes

    model, criterion, optimizer = get_context(args)
    onnx.export(model, next(iter(train_loader))[0], f"output_{title}.onnx")
    return (
        train_model(
            train_loader,
            test_loader,
            model,
            criterion,
            optimizer,
        ),
        train_loader,
        test_loader,
    )


def compare_experiment(*exps):
    for exp in exps:
        args, title = exp
        (x, y, _), _, _ = run_experiment(args, title)
        plt.plot(x, y, label=title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="best")
    plt.show()
