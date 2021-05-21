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





def disparity(imgL,imgR):
    
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16, blockSize=25, P1=0, P2=0) #change block size 
    disparity = stereo.compute(imgL,imgR).astype(np.float32)

    return disparity

def triangulation(disp, data):
    
    """
    solve for triangualation using disparity data and xy coordinate
    """
    
    Q = np.asarray([[1, 0, 0, -disp.shape[1]/2],
                    [0, 1, 0, -disp.shape[0]/2],
                    [0, 0, 0, -7.183351e+02],
                    [0, 0, -1/.54, 0]])

    disp = disp[data[:,1].astype(int), data[:,0].astype(int)]
    flat_disp = abs(disp.flatten())
    new_mat = np.concatenate( ( data.T, flat_disp.reshape(1,len(flat_disp)), np.ones((1,len(flat_disp))) ) ,axis=0)
    world_cordinate = Q.dot(new_mat)

    #normalize#
    world_cordinate = world_cordinate/(np.repeat(world_cordinate[-1,:].reshape(1,len(flat_disp)),4,axis=0)) 
    
    
    return world_cordinate.T

