import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch import functional as F
from torch.utils import data
# from matplotlib import pyplot as plt


class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.linear1 = nn.Linear(50 * 4 * 4, 500)
        self.linear2 = nn.Linear(500, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, batch_data):
        batch_data = self.pool(F.relu(self.conv1(batch_data)))
        batch_data = self.pool(F.relu(self.conv2(batch_data)))
        


if __name__ == '__main__':
    path_to_database = "/mnt/data/Current_Work/Documents/" \
        "Digit_Recognition"

    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    mnist_trainset = datasets.MNIST(path_to_database,
                                    download=False, train=True,
                                    transform=transform_mnist)
    mnist_valset = datasets.MNIST(path_to_database,
                                  download=False, train=False,
                                  transform=transform_mnist)
    trainset_generator = data.DataLoader(mnist_trainset, batch_size=64,
                                         shuffle=True)
    valset_generator = data.DataLoader(mnist_valset, batch_size=64,
                                       shuffle=True)

    # neuralnet = nn.Sequential(nn.Conv2d(1, 20, 5, 1), nn.ReLU(),
    #                           nn.MaxPool2d(2, 2), nn.Conv2d(20, 50, 5, 1),
    #                           nn.ReLU(), nn.MaxPool2d(2, 2),
    #                           nn.Linear(50 * 4 * 4, 500), nn.ReLU(),
    #                           nn.Linear(500, 10), nn.LogSoftmax(dim=1))