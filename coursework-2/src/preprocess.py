import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from common.config import config
from common.utils import get_caltech_101_dataset_dir

dataloader_kwargs = {"batch_size": config.batch_size, "shuffle": True}


class Caltech101ImageDataset(Dataset):
    def __init__(self, image_data, transform=None):
        self.image_data = image_data
        self.transform = transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        path, label = self.image_data[idx]

        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_train_test_data():
    dataset_dir = get_caltech_101_dataset_dir()
    directory_path = Path(dataset_dir)

    train_data = []
    test_data = []
    num_of_classes = 0
    for class_path in directory_path.iterdir():
        num_of_classes += 1

        selected_images = random.sample(
            [image_path for image_path in class_path.iterdir()],
            config.data_count.train + config.data_count.test,
        )
        train_data += [
            (path, class_path.name)
            for path in selected_images[: config.data_count.train]
        ]
        test_data += [
            (path, class_path.name)
            for path in selected_images[config.data_count.train :]
        ]

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = Caltech101ImageDataset(train_data, transform)
    test_dataset = Caltech101ImageDataset(test_data, transform)

    train_loader = DataLoader(train_dataset, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, **dataloader_kwargs)

    return train_dataset, test_dataset, train_loader, test_loader, num_of_classes


def preprocess():
    random.seed(config.seed)
    return get_train_test_data()