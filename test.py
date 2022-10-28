import torch
import torchvision
import torch.optim as optim

from Load_Data import mnistDataSet
from Model import Net,trainModel



if __name__ == "__main__":

    network = Net()
    network_state_dict = torch.load('results/model.pth')
    network.load_state_dict(network_state_dict)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('files/', train=False, download=True,
                         transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(
                             (0.1307,), (0.3081,))
                         ]),target_transform = torchvision.transforms.Lambda(lambda y: torch.tensor(y%2))),
                         batch_size=1, shuffle=True)

    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
