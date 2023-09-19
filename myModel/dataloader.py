import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def fetch_train_data(name, batchsize):
    match name:
        case "FashionMNIST":
            # 70000 28x28 grayscale images
            # in 10 classes with 7000 images per class
            # 60000 training images and 10000 test images
            train_dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())

        case "CIFAR-10":
            # 60000 32x32 colour images
            # in 10 classes with 6000 images per class
            # 50000 training images and 10000 test images
            train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=ToTensor())

        case _:
            print("No data matching name/description")
            return None, None
    
    # dataloader is a wrapper class that supports automatic batching, sampling, shuffling, etc.
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True)

    return train_loader




def fetch_test_data(name, batchsize):
    match name:
        case "FashionMNIST":
            test_dataset = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

        case "CIFAR-10":
            test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=ToTensor())

        case _:
            print("No data matching name/description")
            return None
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=True)

    return test_loader

