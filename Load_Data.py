import torch
import torchvision
import matplotlib.pyplot as plt

class mnistDataSet:
    def __init__(self,batchSize_train,batchSize_test):
        self.loadData_Train(batchSize_train)
        self.loadData_Test(batchSize_test)

    def loadData_Train(self,batchSize_train):
        self.train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]),target_transform = torchvision.transforms.Lambda(lambda y: torch.tensor(y%2))),
                             batch_size=batchSize_train, shuffle=True)
    def loadData_Test(self,batchSize_test):
        self.test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]),target_transform = torchvision.transforms.Lambda(lambda y: torch.tensor(y%2))),
                             batch_size=batchSize_test, shuffle=True)
    def plot_train(self):
        examples = enumerate(self.train_loader)
        batch_idx, (example_data, example_targets) = next(examples)
        print ("Train Batch SIze Shape",example_data.shape)
        fig = plt.figure()
        for i in range(6):
          plt.subplot(2,3,i+1)
          plt.tight_layout()
          plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
          plt.title("Ground Truth: {}".format(example_targets[i]))
          plt.xticks([])
          plt.yticks([])
        fig.savefig("files/train.png")
