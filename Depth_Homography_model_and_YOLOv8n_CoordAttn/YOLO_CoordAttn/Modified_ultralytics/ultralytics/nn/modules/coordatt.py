'''
MIT License

Copyright (c) 2025 Erfan Dilfanian, Qibin (Andrew) Hou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ('CoordinateAttention')

# MobileNetV3-style hard sigmoid (approximates sigmoid using a linear segment, efficient alternative to sigmoid)
class hard_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(hard_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        # Efficient piecewise-linear sigmoid: (ReLU6(x + 3)) / 6
        return self.relu(x + 3) / 6

# MobileNetV3-style hard swish activation (approximate swish: x * sigmoid(x))
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = hard_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

# Coordinate Attention Module
class CoordinateAttention(nn.Module):

    def __init__(self, inp, oup, reduction=32): # inp: number of input channel, oup: number of output channel
        super(CoordinateAttention, self).__init__()
        
        # 1D global pooling along width and height respectively
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        # Shared 1x1 convolution for feature transformation after pooling
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        # Separate 1x1 convolutions to generate attention maps for H and W
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x   # Save input for later multiplication
        
        _,_,h,w = x.size()

        # Generate 1D encodings along vertical and horizontal directions
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate vertical and horizontal features along spatial dimension
        y = torch.cat([x_h, x_w], dim=2)

        # Apply shared transformations
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        # Split the feature back into vertical and horizontal parts
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Generate attention weights
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Apply attention weights along both dimensions
        out = identity * a_w * a_h
        # to see whether Coordinate Attention is being used or not
        # print(">> >> >> CoordinateAttention used")

        return out