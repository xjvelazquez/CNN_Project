import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import pandas as pd
import torchvision
from torchvision import transforms, utils
import os
from sklearn.model_selection import StratifiedKFold
import copy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class loader(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.transform = transform
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.frame.iloc[idx, 0])
        img = Image.open(img_name)
        label = self.frame.iloc[idx, 1]
        
        if self.transform:
            img = self.transform(img)

        return img, label
    
    
class Nnet(nn.Module):
    def __init__(self):
        super(Nnet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 18, 3, stride=1, padding=1, bias=False), 
            nn.ReLU(inplace=True),
            nn.Conv2d(18, 30 , 3, stride=1, padding=1, bias=False), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3),
            nn.Conv2d(30, 35, 3,stride=1, padding=1, bias=False),
            nn.Conv2d(35, 40, 3,stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3),
            nn.Conv2d(40, 45, 3, stride=1, padding=1, bias=False),
            nn.Conv2d(45, 50, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU( inplace=True),
            nn.MaxPool2d(3),
            nn.Conv2d(50, 55, 3, stride=1, padding=1, bias=False),
            nn.Conv2d(55, 60, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU( inplace=True),
            nn.MaxPool2d(3),
        )
        self.fc = nn.Sequential(
            nn.Linear(240,215), # 240, 500
            nn.ReLU( inplace=True),
            #nn.Linear(500,500), # 500, 500
            #nn.ReLU( inplace=True),
            #nn.Linear(230,215), # 500, 500
            #nn.ReLU( inplace=True),
            nn.Linear(215, 201), # 500, 201
            #nn.Softmax()
        )

    def num_flat_features(self, inputs):
        
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        
        for s in size:
            num_features *= s
        
        return num_features

    def forward(self, input):
        x=self.main(input)
        #print('num:', self.num_flat_features(x))
        x=x.view(-1, self.num_flat_features(x))
        
        return self.fc(x)

        
