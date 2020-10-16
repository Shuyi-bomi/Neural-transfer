#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:02:26 2020
project apm598
@author: Shuyi Li
"""
from __future__ import print_function


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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
