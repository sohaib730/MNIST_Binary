import torch
import torchvision
import torch.optim as optim

from Load_Data import mnistDataSet
from Model import Net,trainModel



if __name__ == "__main__":
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    #Initialization
    data = mnistDataSet(batch_size_train,batch_size_test)
    data.plot_train()
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum)



    #Training
    Training = trainModel(network,optimizer,data.train_loader,data.test_loader,n_epochs,log_interval)
    Training.test()
    for epoch in range(1, n_epochs + 1):
        Training.train(epoch)
        Training.test()

    
