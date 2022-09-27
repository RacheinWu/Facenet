import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


def main():
    batch_size = 32
    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)


    cifar_text = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_text = DataLoader(cifar_text, batch_size=batch_size, shuffle=True)

    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)






if __import__ == '_main_':
    main()
