mport os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')

class SelfAttention(nn.Module):

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim
        
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        
        proj_query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, H * W)
        proj_value = self.value_conv(x).view(batch_size, -1, H * W)
        
        
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # resnet
        out = self.gamma * out + x
        return out

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        
        return out

class Generator(nn.Module):
    
    def __init__(self, nz=100, ngf=48, nc=3):  
        super(Generator, self).__init__()
        self.nz = nz
        
        
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 16 * 4 * 4),  
            nn.BatchNorm1d(ngf * 16 * 4 * 4),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )
        
        
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            ResidualBlock(ngf * 8, ngf * 8)  
        )
        
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            ResidualBlock(ngf * 4, ngf * 4)  
        )
        
          
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            ResidualBlock(ngf * 2, ngf * 2),
            
            SelfAttention(ngf * 2)
        )
        
        
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            ResidualBlock(ngf, ngf)  
        )
        
        
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            ResidualBlock(ngf // 2, ngf // 2)  
        )
        
        
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(ngf // 2, ngf // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),
            ResidualBlock(ngf // 4, ngf // 4) 
        )
        
      
        self.final_layers = nn.Sequential(
            
            nn.Conv2d(ngf // 4, ngf // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),
            
            
            nn.Conv2d(ngf // 4, nc, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
      
        if len(input.shape) == 4:
            input = input.view(input.size(0), -1)
        elif len(input.shape) == 2:
            pass
        else:
            raise ValueError(f"Unexpected input shape: {input.shape}")
            
        
        x = self.fc(input)
        x = x.view(x.size(0), -1, 4, 4)
        
        
        x = self.layer1(x)   # 4x4 -> 8x8
        x = self.layer2(x)   # 8x8 -> 16x16
        x = self.layer3(x)   # 16x16 -> 32x32
        x = self.layer4(x)   # 32x32 -> 64x64
        x = self.layer5(x)   # 64x64 -> 128x128
        x = self.layer6(x)   # 128x128 -> 256x256
        x = self.final_layers(x)  
        
        return x
