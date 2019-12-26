import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch import functional as F
from torch.utils import data
# from matplotlib import pyplot as plt


class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 20, 5, 1)
        self.linear = nn.Linear(12 * 12 * 20, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, batch_data):
        batch_data = self.pool(F.relu(self.conv(batch_data)))
        batch_data = batch_data.view(-1, 12 * 12 * 20)
        batch_data = F.relu(self.linear(batch_data))
        return F.log_softmax(batch_data, dim=1)

    def training_pass(self, train_data, epochs, loss_fxn, optimizer):
        for e in range(epochs):
            epoch_loss = 0
            for images, labels in train_data:
                optimizer.zero_grad()
                output = self(images)
                batch_loss = loss_fxn(output, labels)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()

        return epoch_loss

    def test_pass(self, test_data):
        test_results = 0
        total_labels = 0
        with torch.no_grad():
            for images, labels in test_data:
                output = self(images)
                _, prediction = torch.max(output.data, 1)
                test_results += (prediction == labels).sum().item()
                total_labels += labels.size(0)

        return test_results / total_labels


if __name__ == '__main__':
    path_to_database = "/mnt/data/Current_Work/Documents/" \
        "Digit_Recognition"

    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    mnist_trainset = datasets.MNIST(path_to_database,
                                    download=False, train=True,
                                    transform=transform_mnist)
    mnist_testset = datasets.MNIST(path_to_database,
                                   download=False, train=False,
                                   transform=transform_mnist)
    trainset_generator = data.DataLoader(mnist_trainset, batch_size=64,
                                         shuffle=True)
    testset_generator = data.DataLoader(mnist_testset, batch_size=64,
                                        shuffle=True)

    conv_net = ConvNet()

    loss_fxn = nn.NLLLoss()
    optimizer = optim.SGD(conv_net.parameters(), lr=0.01)
    epochs = 7
    test_loss = conv_net.training_pass(trainset_generator, epochs,
                                       loss_fxn, optimizer)
    print(test_loss / len(trainset_generator))
    print(100 * conv_net.test_pass(testset_generator))
