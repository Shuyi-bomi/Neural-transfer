#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:02:26 2020
project apm598
@author: Shuyi Li
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import os

#plot
def preprocess(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0) 
    image = unloader(image)
    return image
outputst = preprocess(outputst)
outputct = preprocess(outputct)
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(4, 4)
axes[0].imshow(outputst)
axes[0].set_title("Style Transfer")
axes[1].imshow(outputct)
axes[1].set_title("Content Transfer")

plt.ioff()
plt.show()


# plot loss
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(8, 4)
axes[0].plot(np.arange(320),losspro[:320,0],'--',label='Content Loss')
axes[0].plot(np.arange(320),losspro[:320,1],'-.',label='Style Loss')
axes[0].plot(np.arange(320),losspro[:320,2],':',label='Total Loss')
axes[0].set_title("Style Transfer")
axes[0].legend()
axes[1].plot(np.arange(320),losspro[:320,3],'--',label='Content1 Loss(shepherd)')
axes[1].plot(np.arange(320),losspro[:320,4],'-.',label='Content2 Loss(pomeranian)')
axes[1].plot(np.arange(320),losspro[:320,5],':',label='Total Loss')
axes[1].set_title("Content Transfer")
axes[1].legend()

