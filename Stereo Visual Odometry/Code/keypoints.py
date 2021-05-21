import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys
import skimage.io
from scipy.spatial import distance
import scipy
import random
from scipy.optimize import least_squares
from math import cos, sin





def get_sift_data(img):
    #sift = cv2.SIFT_create()
    sift = cv2.ORB_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

# def get_sift_data(imgL):

#     TILE_H = 10
#     TILE_W = 20
#     #orb = cv2.ORB_create()
#     orb = cv2.SIFT_create()

#     # 20x10 (wxh) tiles for extracting less features from images
#     H, W = imgL.shape
#     kp = []
#     des = []
#     idx = 0
#     for y in range(0, H, TILE_H):
#         for x in range(0, W, TILE_W):

#             imPatch = imgL[y:y + TILE_H, x:x + TILE_W]
#             kt = orb.detect(imPatch,None)
#             keypoint,description = orb.compute(imPatch, kt)
#             keypoint = np.asarray(keypoint)
#             description = np.asarray(description)

#             for pt in keypoint:
#                 pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

#             try:   
#                 if (len(keypoint) > 20):

#                     kp_list = np.asarray([i.response for i in keypoint])
#                     max_idx = kp_list.argsort()[-20:]

#                     for kpt,desc in zip(keypoint[max_idx],description[max_idx]):
#                         kp.append(kpt)
#                         des.append(desc)
#                 else:
#                     for kpt,desc in zip(keypoint,description):
#                         kp.append(kpt)
#                         des.append(desc)
#             except:
#                 pass

    
#     return kp, des



def get_best_matches(img1, img2, num_matches):
    kp1, des1 = get_sift_data(img1)
    kp2, des2 = get_sift_data(img2)
    kp1, kp2 = np.array(kp1), np.array(kp2)
    
    # Find distance between descriptors in images
    dist = scipy.spatial.distance.cdist(des1, des2, 'sqeuclidean')
    
    # Write your code to get the matches according to dist
    # <YOUR CODE>


    # des 1 is in row
    # des 2 is in col

    min_row = np.argmin(dist,axis=1)
    min_col = np.argmin(dist,axis=0)

    #use below idx as input to min_row
    match_row_idx = [i for i in range(0,len(min_row)) if min_col[min_row[i]]==i]
    #match_row_idx has feature id from des1 and min_row has corresponding feature from des2

    kp_1_loc = cv2.KeyPoint_convert(kp1[match_row_idx])
    kp_2_loc = cv2.KeyPoint_convert(kp2[min_row[match_row_idx]])
    data = np.concatenate((kp_1_loc,kp_2_loc),axis=1)
    
    return data





