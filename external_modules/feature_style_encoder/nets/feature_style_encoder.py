import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torchvision import models, utils

from arcface.iresnet import *


# class fs_encoder_v2(nn.Module):
#     def __init__(self, n_styles=18, opts=None, residual=False, use_coeff=False, resnet_layer=None, video_input=False, f_maps=512, stride=(1, 1)):
#         super(fs_encoder_v2, self).__init__()  

#         resnet50 = iresnet50()
#         resnet50.load_state_dict(torch.load(opts.arcface_model_path))

#         # input conv layer
#         if video_input:
#             self.conv = nn.Sequential(
#                 nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
#                 *list(resnet50.children())[1:3]
#             )
#         else:
#             self.conv = nn.Sequential(*list(resnet50.children())[:3])
        
#         # define layers
#         self.block_1 = list(resnet50.children())[3] # 15-18
#         self.block_2 = list(resnet50.children())[4] # 10-14
#         self.block_3 = list(resnet50.children())[5] # 5-9
#         self.block_4 = list(resnet50.children())[6] # 1-4
        

#         self.avg_pool = nn.AdaptiveAvgPool2d((3,3))
#         self.styles = nn.ModuleList()
#         for i in range(n_styles):
#             self.styles.append(nn.Linear(960 * 9, 512))

#         self.idx_k = int(opts.idx_k)
#         if  self.idx_k  in [4,5]:
#             self.feat_size = 16
#             self.feat_ch = 512
#             self.in_feat = 256
#         elif self.idx_k in [6,7]:
#             self.feat_size = 32
#             self.feat_ch = 512
#             self.in_feat = 128
#         elif self.idx_k in [8,9]:
#             self.feat_size = 64
#             self.feat_ch = 512
#             self.in_feat = 64
#         elif self.idx_k in [2,3]:
#             self.feat_size = 16
#             self.feat_ch = 512
#             self.in_feat = 512
#         self.content_layer = nn.Sequential(
#             nn.BatchNorm2d(self.in_feat, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.Conv2d(self.in_feat, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
#             nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.PReLU(num_parameters=512),
#             nn.Conv2d(512, 512, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
#             nn.BatchNorm2d(self.feat_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#     def forward(self, x):
#         latents = []
#         features = []
#         x = self.conv(x)
#         x = self.block_1(x)
#         # print("block1", x.shape)  # torch.Size([batch_size, 64, 128, 128])
#         if self.idx_k in [8,9]:
#             content = self.content_layer(x) # size = (batch_size, 512, 64, 64)
#         features.append(self.avg_pool(x))
#         x = self.block_2(x)

#         # print("block2", x.shape) # torch.Size([batch_size, 128, 64, 64])
#         if self.idx_k in [6,7]:
#             content = self.content_layer(x) # size = (batch_size, 512, 32, 32)
#         features.append(self.avg_pool(x))
#         x = self.block_3(x)
 
#         # print("block3", x.shape) # torch.Size([batch_size, 256, 32, 32])
#         if  self.idx_k  in [4,5]:
#             content = self.content_layer(x) # size = (batch_size, 512, 16, 16)
#         features.append(self.avg_pool(x))
#         x = self.block_4(x)
#         features.append(self.avg_pool(x))
#         x = torch.cat(features, dim=1)
#         x = x.view(x.size(0), -1)
#         for i in range(len(self.styles)):
#             latents.append(self.styles[i](x))
#         out = torch.stack(latents, dim=1)
#         return out, content




class fs_encoder_v2(nn.Module):
    def __init__(self, n_styles=18, opts=None, residual=False, use_coeff=False, resnet_layer=None, video_input=False, f_maps=512, stride=(1, 1)):
        super(fs_encoder_v2, self).__init__()  

        resnet50 = iresnet50()
        resnet50.load_state_dict(torch.load(opts.arcface_model_path))

        # input conv layer
        if video_input:
            self.conv = nn.Sequential(
                nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                *list(resnet50.children())[1:3]
            )
        else:
            self.conv = nn.Sequential(*list(resnet50.children())[:3])
        
        # define layers
        self.block_1 = list(resnet50.children())[3] # 15-18
        self.block_2 = list(resnet50.children())[4] # 10-14
        self.block_3 = list(resnet50.children())[5] # 5-9
        self.block_4 = list(resnet50.children())[6] # 1-4
        

        self.avg_pool = nn.AdaptiveAvgPool2d((3,3))
        self.styles = nn.ModuleList()
        for i in range(n_styles):
            self.styles.append(nn.Linear(960 * 9, 512))

        self.idx_k = int(opts.idx_k)
        stride = stride
        if  self.idx_k in [2,3]:
            self.feat_size = 16
            self.feat_ch = 512
            self.in_feat = 512
        elif  self.idx_k  in [4,5]:
            self.feat_size = 16
            self.feat_ch = 512
            self.in_feat = 256
        elif self.idx_k in [6,7]:
            self.feat_size = 32
            self.feat_ch = 512
            self.in_feat = 128
        elif self.idx_k in [8,9]:
            self.feat_size = 64
            self.feat_ch = 512
            self.in_feat = 64
        elif self.idx_k in [10,11]:
            self.feat_size = 128
            self.feat_ch = 256
            self.in_feat = 64
        elif self.idx_k in [12,13]:
            stride = (1, 1)
            self.feat_size = 256
            self.feat_ch = 128
            self.in_feat = 64
        elif self.idx_k in [14,15]:
            stride = (1, 1)
            self.feat_size = 512
            self.feat_ch = 64
            self.in_feat = 64
        
        self.content_layer = nn.Sequential(
            nn.BatchNorm2d(self.in_feat, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(self.in_feat, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.PReLU(num_parameters=512),
            
            nn.Conv2d(512, self.feat_ch, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(self.feat_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        )

    def forward(self, x):
        latents = []
        features = []
        x = self.conv(x) # size = (batch_size, 64, 256, 256)
        # print(x.shape)
        # import pdb; pdb.set_trace()
        if self.idx_k in [10,11,12,13,14,15]:
            content = self.content_layer(x) # size = (batch_size, 256, 128, 128)
        x = self.block_1(x)  # torch.Size([batch_size, 64, 128, 128])
        
        if self.idx_k in [8,9]:
            content = self.content_layer(x) # size = (batch_size, 512, 64, 64)
        features.append(self.avg_pool(x))
        x = self.block_2(x)

        # print("block2", x.shape) # torch.Size([batch_size, 128, 64, 64])
        if self.idx_k in [6,7]:
            content = self.content_layer(x) # size = (batch_size, 512, 32, 32)
        features.append(self.avg_pool(x))
        x = self.block_3(x)
 
        # print("block3", x.shape) # torch.Size([batch_size, 256, 32, 32])
        if  self.idx_k  in [4,5]:
            content = self.content_layer(x) # size = (batch_size, 512, 16, 16)
        features.append(self.avg_pool(x))
        x = self.block_4(x)
        features.append(self.avg_pool(x))
        x = torch.cat(features, dim=1)
        x = x.view(x.size(0), -1)
        for i in range(len(self.styles)):
            latents.append(self.styles[i](x))
        out = torch.stack(latents, dim=1)
        return out, content

    
class StyleMod(nn.Module):
    def __init__(self, n_styles=18):
        super(StyleMod, self).__init__()  

        self.style_layers = nn.ModuleList()
        for i in range(n_styles):
            self.style_layers.append(LinearLayer())

    def forward(self, x):
        latents = []
        for i, style_layer in enumerate(self.style_layers):
            latents.append(style_layer(x[:,i,:]))
        out = torch.stack(latents, dim=1)
        return out

class LinearLayer(nn.Module):
    def __init__(self):
        super(LinearLayer, self).__init__()
        
        layer = []
        layer.append(nn.Linear(512, 512))
        layer.append(nn.PReLU(num_parameters=512))
        layer.append(nn.Linear(512, 512))

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)