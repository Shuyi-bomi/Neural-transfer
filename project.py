#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:02:26 2020
project apm598
@author: apple
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from PIL import Image
#from matplotlib import pyplot
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import dlib
import cv2
import os
from imutils import face_utils


detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')

def loadim(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
img = loadim('pomeranian-900212_1280.jpg')
img2 = loadim('australian-shepherd-3237735_1280.jpg')
#img2 = cv2.resize(img2, dsize=(902,1280), fx=0.5, fy=0.5)

#filename, ext = os.path.splitext(os.path.basename(img_path1))

fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(5, 5)
axes[0].imshow(img)
axes[1].imshow(img2)

# detection 
# 0 indicates #of upsample,make everything bigger and allow us to detect more faces
dets = detector(img, upsample_num_times=0)
dets2 = detector(img2, upsample_num_times=0)

print(dets2)

img_result = img.copy()
img_result2 = img2.copy()


def plotdet(dets,img_result,loc=[]):
    
    for i, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
        loc.append(d.rect.left())
        loc.append(d.rect.top())
        loc.append(d.rect.right())
        loc.append(d.rect.bottom())
        x1, y1 = d.rect.left(), d.rect.top()
        x2, y2 = d.rect.right(), d.rect.bottom()
        cv2.rectangle(img_result, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=(255,0,0), lineType=cv2.LINE_AA)
        return img_result,loc
img_result,loc = plotdet(dets, img_result, loc=[]) 
img_result2,loc2 = plotdet(dets2, img_result2, loc=[]) 


# fig=plt.figure(figsize=(5, 5))
# plt.imshow(img_result2) 
# fig.savefig('imgde.png', dpi=200, bbox_inches="tight")


#if you see the rectangular not complete , comment/delete it (latex add)
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(5, 5)
axes[0].imshow(img_result)
axes[1].imshow(img_result2)


loc = np.array(loc).reshape((-1,4)) 
loc2 = np.array(loc2).reshape((-1,4)) 

    
def crop_faces(im, size, loc):
  """ Returns cropped faces from an image """
  faces = []
  locations = loc
  for location in locations:
    y1, x1, y2, x2 = location
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    #if img.shape[0]>img.shape[1]:
    im_cropped = im[x1:x2, y1:y2]
    im_cropped_resized = cv2.resize(im_cropped, dsize=size)
    faces.append(im_cropped_resized)
  return faces
imac = crop_faces(img, (256,256), loc)[0]
imac2 = crop_faces(img2, (256,256), loc2)[0]

fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(8, 8)
axes[0].imshow(imac)
axes[1].imshow(imac2)

#fig=plt.figure(figsize=(5, 5))
#plt.imshow(img_result2) 
#fig.savefig('imgcrop.jpg', dpi=200, bbox_inches="tight")


#style/content transfer
imsize = 512 if torch.cuda.is_available() else 128  

# loader1 = transforms.Compose([
#     transforms.Resize((imsize,196)) ])  
# loader2 = transforms.Compose([
#     transforms.ToTensor()])  
loader = transforms.Compose([
    #transforms.Resize((imsize,196)), 
    transforms.ToTensor()])  

def image_loader(image_name):
    #image = Image.open(image_name)
    image = loader(image_name).unsqueeze(0)
    return image.to(device, torch.float)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
style_img = image_loader(imac) ##pom style 
content_img = image_loader(imac2)
#style_img.size()
#content_img.size()

# assert style_img.size() == content_img.size(),\
#     "You have to to import style and content images of the same size"
    
unloader = transforms.ToPILImage() 
plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone() 
    image = image.squeeze(0)      
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 

# plt.figure()
# imshow(style_img, title='Style Image')

# plt.figure()
# imshow(content_img, title='Content Image')

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
    
def gram_matrix(input):
    a, b, c, d = input.size()  

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t()) 

    return G.div(a * b * c * d)    
    
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input    
cnn = models.vgg19(pretrained=True).features.to(device).eval()   
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = (mean).view(-1, 1, 1)
        self.std = (std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std    
    
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)        
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            # For content loss, define both loss as content loss
            if len(style_layers)>1:
                style_loss = StyleLoss(target_feature)
            else:
                style_loss = ContentLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
            
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses    
    
input_img = content_img.clone()
# plt.figure()
# imshow(input_img, title='Input Image')    
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer 
hyperP = dict()
hyperP['num_steps'] = 300
hyperP['style_weight'] = 1000000
hyperP['content_weight'] = 1

losspro = np.zeros((hyperP['num_steps']+100, 6))
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=hyperP['num_steps'],
                       style_weight=hyperP['style_weight'], content_weight=hyperP['content_weight'], st='T'):
   
    model, style_losses, content_losses = get_style_model_and_losses(cnn,normalization_mean, 
      normalization_std, style_img, content_img, content_layers_default, style_layers_default)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()
            if st=='T':
                losspro[run[0],0] = content_score.item()
                losspro[run[0],1] = style_score.item()
                losspro[run[0],2] = loss.item()
            else:
                losspro[run[0],3] = content_score.item()
                losspro[run[0],4] = style_score.item()
                losspro[run[0],5] = loss.item()

            run[0] += 1
            if run[0] % 100 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            

            return style_score + content_score

        optimizer.step(closure)


    input_img.data.clamp_(0, 1)
    

    return input_img
torch.manual_seed(1)
outputst = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img, num_steps=hyperP['num_steps'],
                            style_weight=hyperP['style_weight'], content_weight=hyperP['content_weight'] )
style_layers_default = ['conv_4']
input_img = content_img.clone()
hyperP['style_weight'] = 1.5
torch.manual_seed(1)
outputct = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img, num_steps=hyperP['num_steps'],
                            style_weight=hyperP['style_weight'], content_weight=hyperP['content_weight'],st='F' )

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




    
    
    
