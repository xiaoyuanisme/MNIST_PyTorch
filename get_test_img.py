from  torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import  numpy as np
from PIL import Image
import torch
from testinf import  simpleInfer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
test_transform = transforms.Compose([np.array])
test_dataset = MNIST(root='data', train=False, transform=test_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)

for imgs,label in test_loader:
    for i,img in enumerate( imgs.numpy()):
        Image.fromarray(img).save("imgs/%d.png"%i)
    break

infer = simpleInfer()
inp = [ np.array(Image.open("imgs/%d.png"%i))  for i in range(10)]
ans = infer.infer(inp)
print(ans)

from train import Net,test_transform

net = Net()


print("Model summary")
from torchsummary import summary
summary(net,(1,28,28))


from thop import profile
flops, params = profile(net, inputs=(torch.randn(1, 1, 28, 28), ))

print("flops:",flops)
print("params:",params)