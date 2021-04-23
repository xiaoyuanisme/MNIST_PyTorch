
import torch
from train import Net,test_transform
import numpy as np
from PIL import Image

class simpleInfer(object):
    def __init__(self,model_path = "model.pt"):
        self.model = self.load_model(model_path)
        self.transform = test_transform
    def load_model(self,model_path:str):
        model = Net()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model
    def infer(self,inp:list):
        ans = []
        for img in inp:
            img = self.transform(img)
            img = img.unsqueeze(0)
            out = self.model(img)
            ans.append(out.argmax(1)[0].item())
        return ans 
