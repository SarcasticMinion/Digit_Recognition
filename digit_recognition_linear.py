import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils import data
# from matplotlib import pyplot as plt


class BasicNN(nn.Module):

    def __init__(self, optimizer, loss_fxn, neural_net, trainset):

        super().__init__()
        self.optimizer = optimizer
        self.loss_fxn = loss_fxn
        self.net = neural_net
        self.trainset = trainset

    def one_epoch(self, images, labels):

        images = images.view(images.shape[0], -1)
        self.optimizer.zero_grad()
        output = self.net(images)
        loss = self.loss_fxn(output, labels)
        loss.backward()
        self.optimizer.step()

        return output, loss

    def training_pass(self, epochs):
        self.train_loss = 0
        for e in range(epochs):
            for images, labels in self.trainset:
                loss = self.one_epoch(images, labels)[1]
                self.train_loss += loss.item()

    def validation_pass(self, valset):
        test_results = 0
        total_labels = 0
        with torch.no_grad():
            for images, labels in valset:
                images = images.view(images.shape[0], -1)
                output = self.net(images)
                _, prediction = torch.max(output.data, 1)
                test_results += (prediction == labels).sum().item()
                total_labels += labels.size(0)
        self.accuracy = test_results / total_labels


if __name__ == '__main__':

    transform_mnist = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])
    mnist_trainset = datasets.MNIST('/home/minion/Documents/Machine',
                                    download=False, train=True,
                                    transform=transform_mnist)
    mnist_valset = datasets.MNIST('/home/minion/Documents/Machine',
                                  download=False, train=False,
                                  transform=transform_mnist)
    trainset_generator = data.DataLoader(mnist_trainset, batch_size=64,
                                         shuffle=True)
    valset_generator = data.DataLoader(mnist_valset, batch_size=64,
                                       shuffle=True)

    input_length = 28 * 28
    hidden_layers = [256, 128, 64]
    output_length = 10

    # temp = iter(mnist_trainset)
    # images, labels = next(temp)
    # plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
    # plt.show()
    neuralnet = nn.Sequential(nn.Linear(input_length, hidden_layers[0]),
                              nn.ReLU(), nn.Linear(hidden_layers[0], hidden_layers[1]),
                              nn.ReLU(), nn.Linear(hidden_layers[1], hidden_layers[2]),
                              nn.ReLU(), nn.Linear(hidden_layers[2], output_length),
                              nn.LogSoftmax(dim=1))

    loss_fxn = nn.NLLLoss()
    optimizer = optim.SGD(neuralnet.parameters(), lr=0.01)

    my_training = BasicNN(optimizer, loss_fxn, neuralnet, trainset_generator)
    my_training.training_pass(35)
    my_training.validation_pass(valset_generator)
    print(my_training.train_loss / len(trainset_generator))
    print(100 * my_training.accuracy)
