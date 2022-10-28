import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class trainModel:
    def __init__(self,network,optimizer,train_loader,test_loader,n_epochs,log_interval):
        self.network = network
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
                                            #For Display
        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
        self.log_interval = log_interval

    def train(self,epoch):
        self.network.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(self.train_loader.dataset),
                100. * batch_idx / len(self.train_loader), loss.item()))
                self.train_losses.append(loss.item())
                self.train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(self.train_loader.dataset)))
                torch.save(self.network.state_dict(), 'results/model.pth')
                torch.save(self.optimizer.state_dict(), 'results/optimizer.pth')

    def test(self):
        self.network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(self.test_loader.dataset),
        100. * correct / len(self.test_loader.dataset)))
