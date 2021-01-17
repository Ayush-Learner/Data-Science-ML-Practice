import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
import sys
import random
import os, numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
# from skimage.transform import resize
from scipy.sparse import csr_matrix
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt


train_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.RandomHorizontalFlip(p=1),
                                       ])

copy_transforms = transforms.Compose([transforms.ToTensor()
                                       ])

#Second Try
class RoadSequenceDatasetList(data.Dataset):

    def __init__(self, file_path, transforms_1,flag):
        
        file_path=file_path+"/"+flag
        self.img_list = os.listdir(file_path)
        #print(self.img_list)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms_1
        self.file_path=file_path
        print("epsilon_")
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        
        data = []
        label = []
        path=self.file_path+'/'+self.img_list[idx]
        
        epsilon = 0.2
        #print("epsilon")
        """
        for i in os.listdir(path):
            
            if np.random.rand() >= epsilon:
                if self.transforms(Image.open(os.path.join(path,i))).size()[0]==3:
                    
                    data.append(torch.unsqueeze(self.transforms(Image.open(os.path.join(path,i))), dim=0))
                else:
                    #stack them and transform them all
                    label = Image.open(os.path.join(path,i))
                    label = torch.squeeze(self.transforms(label))
            else :
                self.transforms = copy_transforms
                if self.transforms(Image.open(os.path.join(path,i))).size()[0]==3:
                    self.transforms = train_transforms
                    data.append(torch.unsqueeze(self.transforms(Image.open(os.path.join(path,i))), dim=0))
                else:
                    self.transforms = train_transforms
                    #stack them and transform them all
                    label = Image.open(os.path.join(path,i))
                    label = torch.squeeze(self.transforms(label))
                self.transforms = copy_transforms
                             
        """
        
            
        if np.random.rand() >= epsilon:
            for i in os.listdir(path):
                
                if self.transforms(Image.open(os.path.join(path,i))).size()[0]==3:
                    
                    data.append(torch.unsqueeze(self.transforms(Image.open(os.path.join(path,i))), dim=0))
                else:
                    #stack them and transform them all
                    label = Image.open(os.path.join(path,i))
                    label = torch.squeeze(self.transforms(label))
        else :
            for i in os.listdir(path):
                self.transforms = copy_transforms
                if self.transforms(Image.open(os.path.join(path,i))).size()[0]==3:
                    self.transforms = train_transforms
                    data.append(torch.unsqueeze(self.transforms(Image.open(os.path.join(path,i))), dim=0))
                else:
                    self.transforms = train_transforms
                    #stack them and transform them all
                    label = Image.open(os.path.join(path,i))
                    label = torch.squeeze(self.transforms(label))
                self.transforms = copy_transforms
        
        
        
        #ind = [i for i in range(0,20,2)]
        ind = [i for i in range(0,20,4)]
        #print("cnc")
        
        
        
        data = torch.cat(data, 0)
        data = data[ind]
        sample = {'data': data, 'label': label}
        return sample