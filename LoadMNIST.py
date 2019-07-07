import torch
from torchvision import datasets, transforms
import torchvision.datasets as datasets


def LoadMNIST(batch_size=32):
    root = '../../data'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = datasets.MNIST(root=root, train=True, transform=transform, download=True)
    test_set = datasets.MNIST(root=root, train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)

    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)

    return train_loader, None, test_loader
