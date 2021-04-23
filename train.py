
import torch
import torch.optim as optim
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from  torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm 

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,16,3,padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16,32,3,padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,3,padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,3,padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc =  nn.Sequential(
            # nn.Dropout(),
            nn.Linear(128,512),
            nn.Linear(512,10)
        )
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

# https://zhuanlan.zhihu.com/p/54527197
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

if __name__ == "__main__":

    batch_size = 64
    MAX_EPOCH = 60
    # device = torch.device("cpu")
    device = torch.device("cuda:0")

    train_dataset = MNIST(root='data', train=True, transform=train_transform,  download=True)
    test_dataset = MNIST(root='data', train=False, transform=test_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    net = Net()

    net.to(device)
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(MAX_EPOCH):

        net.train()
        train_loss = 0.0
        count = 0
        correct = 0
        for imgs,labels in tqdm(train_loader):
            imgs,labels = imgs.to(device), labels.to(device) 
            optimizer.zero_grad()

            outputs = net(imgs)

            loss = criterion(outputs, labels)
            loss.backward()

            pre = outputs.argmax(1)
            correct += (pre == labels).sum().item() 
            count = count + labels.shape[0]
            train_loss += (loss.item() * labels.shape[0])
            optimizer.step()
        train_loss = train_loss / count
        print("Epoch[%d/%d] , Train Loss: %.4f, Accuracy:%.3f%%(%d/%d)"%(epoch + 1,MAX_EPOCH,train_loss,(100.0 * correct / count ),correct,count))


        net.eval()
        test_loss = 0.0
        count = 0
        correct = 0
        for imgs,labels in test_loader:
            imgs,labels = imgs.to(device), labels.to(device) 

            outputs = net(imgs)

            loss = criterion(outputs, labels)

            pre = outputs.argmax(1)
            correct += (pre == labels).sum().item() 
            count = count + labels.shape[0]
            test_loss += (loss.item() * labels.shape[0])

        test_loss = test_loss / count
        print("Epoch[%d/%d] , Test Loss: %.4f, Accuracy:%.3f%%(%d/%d)"%(epoch + 1,MAX_EPOCH,test_loss,(100.0 * correct / count ),correct,count))
        torch.save(net.state_dict(),"model.pt")
