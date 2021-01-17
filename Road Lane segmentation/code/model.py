import torch
from utils import *
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class_num=2

class SegNet_ConvLSTM(nn.Module):
    def __init__(self):
        super(SegNet_ConvLSTM,self).__init__()
        self.vgg16_bn = models.vgg16_bn(pretrained=True).features
        self.relu = nn.ReLU(inplace=True)
        self.index_MaxPool = nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.index_UnPool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # net struct
        self.conv1_block = nn.Sequential(self.vgg16_bn[0],  # conv2d(3,64,(3,3))
                                         self.vgg16_bn[1],  # bn(64,eps=1e-05,momentum=0.1,affine=True)
                                         self.vgg16_bn[2],  # relu(in_place)
                                         self.vgg16_bn[3],  # conv2d(3,64,(3,3))
                                         self.vgg16_bn[4],  # bn(64,eps=1e-05,momentum=0.1,affine=True)
                                         self.vgg16_bn[5]   # relu(in_place)
                                         )
        self.conv2_block = nn.Sequential(self.vgg16_bn[7],
                                         self.vgg16_bn[8],
                                         self.vgg16_bn[9],
                                         self.vgg16_bn[10],
                                         self.vgg16_bn[11],
                                         self.vgg16_bn[12]
                                         )
        self.conv3_block = nn.Sequential(self.vgg16_bn[14],
                                         self.vgg16_bn[15],
                                         self.vgg16_bn[16],
                                         self.vgg16_bn[17],
                                         self.vgg16_bn[18],
                                         self.vgg16_bn[19],
                                         self.vgg16_bn[20],
                                         self.vgg16_bn[21],
                                         self.vgg16_bn[22]
                                         )
        self.conv4_block = nn.Sequential(self.vgg16_bn[24],
                                         self.vgg16_bn[25],
                                         self.vgg16_bn[26],
                                         self.vgg16_bn[27],
                                         self.vgg16_bn[28],
                                         self.vgg16_bn[29],
                                         self.vgg16_bn[30],
                                         self.vgg16_bn[31],
                                         self.vgg16_bn[32]
                                         )
        self.conv5_block = nn.Sequential(self.vgg16_bn[34],
                                         self.vgg16_bn[35],
                                         self.vgg16_bn[36],
                                         self.vgg16_bn[37],
                                         self.vgg16_bn[38],
                                         self.vgg16_bn[39],
                                         self.vgg16_bn[40],
                                         self.vgg16_bn[41],
                                         self.vgg16_bn[42]
                                         )

        self.upconv5_block = nn.Sequential(
                                           nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1)),
                                           nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
                                           self.relu,
                                           nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1)),
                                           nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
                                           self.relu,
                                           nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
                                           nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
                                           self.relu,
                                           )
        self.upconv4_block = nn.Sequential(
                                           nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1)),
                                           nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
                                           self.relu,
                                           nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1)),
                                           nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
                                           self.relu,
                                           nn.Conv2d(512, 256, kernel_size=(3, 3), padding=(1, 1)),
                                           nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
                                           self.relu,
                                           )
        self.upconv3_block = nn.Sequential(
                                           nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1,1)),
                                           nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
                                           self.relu,
                                           nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1,1)),
                                           nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
                                           self.relu,
                                           nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1)),
                                           nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
                                           self.relu,
                                           )
        self.upconv2_block = nn.Sequential(
                                           nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1,1)),
                                           nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
                                           self.relu,
                                           nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1,1)),
                                           nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
                                           self.relu
                                           )
        self.upconv1_block = nn.Sequential(
                                           nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1,1)),
                                           nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
                                           self.relu,
                                           nn.Conv2d(64, class_num, kernel_size=(3, 3), padding=(1,1)),
                                           )
        self.convlstm = ConvLSTM(input_size=(4,8),
                                 input_dim=512,
                                 hidden_dim=[512, 512],
                                 kernel_size=(3,3),
                                 num_layers=2,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)
    def forward(self, x):
        #print("inside forward")
        x = torch.unbind(x, dim=1)
        data = []
        for item in x:
            f1, idx1 = self.index_MaxPool(self.conv1_block(item))
            f2, idx2 = self.index_MaxPool(self.conv2_block(f1))
            f3, idx3 = self.index_MaxPool(self.conv3_block(f2))
            f4, idx4 = self.index_MaxPool(self.conv4_block(f3))
            f5, idx5 = self.index_MaxPool(self.conv5_block(f4))
            data.append(f5.unsqueeze(0))
        data = torch.cat(data, dim=0)
        lstm, _ = self.convlstm(data)
        test = lstm[0][-1,:,:,:,:]
        up6 = self.index_UnPool(test,idx5)
        up5 = self.index_UnPool(self.upconv5_block(up6), idx4)
        up4 = self.index_UnPool(self.upconv4_block(up5), idx3)
        up3 = self.index_UnPool(self.upconv3_block(up4), idx2)
        up2 = self.index_UnPool(self.upconv2_block(up3), idx1)
        up1 = self.upconv1_block(up2)
        #print("output shape is", up1.size())
        #print("Exiting forward")
        return F.log_softmax(up1, dim=1)