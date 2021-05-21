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
from triangulation_disparity import disparity, triangulation
from optimization import cost_func


def ransac(data, disp_c, disp_p, loop, Threshold,T):
    """
    write your ransac code to find the best model, inliers, and residuals
    """
    
    # kp1 is imgL_p
    # kp2 is imgL_c
     
    # <YOUR CODE>
    

    

    loop = loop
    Threshold = Threshold
    
    max_count = -1
    
    for i in range(loop):
        
        rnd_idx = random.sample(range(0,len(data)),6)
        U = range(0,len(data))
        comp_rnd_idx = list(set(U) - set(rnd_idx))
        

        H,cost,tri_p,tri_c = compute_motion(data[rnd_idx], disp_c, disp_p,T)
        err = compute_error(data[comp_rnd_idx], disp_c, disp_p,H)
        
        
        #print(H)
        #print(err.shape)
        err = err.reshape((int(len(err)/4),4))
        #print(err.shape)
        err = np.sum(err**2,axis=1)**.5
        err = err/2
    
        
        count = sum(np.where(err<Threshold,1,0))
        #print(count)
        if max_count < count:
            max_count = count
            final_pose = H
            best_model_errors = err
            matching_point = data[rnd_idx]
            #print(best_model_errors.shape)
            #print("############Best Count is##############")
            #print(max_count)
            
    return final_pose, max_count, np.mean(best_model_errors),np.median(best_model_errors), matching_point,cost,tri_p,tri_c


def compute_motion(data, disp_c, disp_p,T):
    """
    solve non linear equation
    """
    threshold = 30
    
    ###Calculating World Co-ordinates####
    ####Triangulation###########
    tri_c = triangulation(disp_c,data[:,2:])
    tri_p = triangulation(disp_p,data[:,:2])
    
    ###function to reject feature with certain disparity value###
    
    cond1 = (abs(tri_c[:,2])< threshold).astype('uint8')
    cond2 = (abs(tri_p[:,2])< threshold).astype('uint8')
    cond = (cond1*cond2).astype('bool')
    
    tri_c = tri_c[cond]
    tri_p = tri_p[cond]
    #print("tri_c",tri_c[np.isnan(tri_c)])
    #print("tri_p",tri_p[np.isnan(tri_p)])
    
    """
    print("tri_c",tri_c[np.isnan(tri_c)])
    print("###################################")
    print("Triangulated world coordinates")
    print(tri_c)
    print("###################################")
    print("Disparity of each pixel")
    print(disp_c[data[:,3].astype(int), data[:,2].astype(int)])
    print("###################################")
    print("Image pixel")
    print(data[:,2:])

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    print("tri_p",tri_p[np.isnan(tri_p)])
    print("###################################")
    print("Triangulated world coordinates")
    print(tri_p)
    print("###################################")
    print("Disparity of each pixel")
    print(disp_p[data[:,1].astype(int), data[:,0].astype(int)])
    print("###################################")
    print("Image pixel")
    print(data[:,:2])
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    """
    
    # kp1 is imgL_p
    # kp2 is imgL_c
    
    ####Homogeneous Image Coordinate#####
    kp1 = data[cond][:,:2]
    kp2 = data[cond][:,2:]
    
    kp1 = np.concatenate((kp1,np.ones((len(kp1),1))),axis=1)
    kp2 = np.concatenate((kp2,np.ones((len(kp2),1))),axis=1)
    
    """
    Using world coordinates, image co-ordinates and Projection matric calculate Motion matrix T
    """
    ####Projection Matrix######
#     P = np.asarray([[7.183351e+02, 0.000000e+00, 6.003891e+02, 4.450382e+01],
#                     [0.000000e+00, 7.183351e+02, 1.815122e+02, -5.951107e-01], 
#                     [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.616315e-03]])



    P = np.asarray([[7.183351e+02, 0.000000e+00, 6.003891e+02, 0],
                [0.000000e+00, 7.183351e+02, 1.815122e+02, 0], 
                [0.000000e+00, 0.000000e+00, 1.000000e+00, 0]])

####Grayscale camera matrix
#     P = np.asarray([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00], 
#                     [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00],
#                     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])
    
    
    
    ###Code###
    """
    Cost function = (image_cordinate_p - P*T*world_cordinate_c)^2 + (image_cordinate_c - P*(T_inverse)*world_cordinate_p)^2
    
    sum above for all features
    
    """

    optRes = least_squares(cost_func, T, method='lm', max_nfev=2000,args=(tri_p, tri_c, kp1, kp2, P))
    T = optRes.x
    cost = optRes.cost/len(kp1[:,0])
    #print("cost is",cost)

    return T,cost,tri_p, tri_c


def compute_error(data, disp_c, disp_p,T):
    """
    solve non linear equation
    """
    
    
    ###Calculating World Co-ordinates####
    ####Triangulation###########
    tri_c = triangulation(disp_c,data[:,2:])
    tri_p = triangulation(disp_p,data[:,:2])
    
    ###function to reject feature with certain disparity value###
    threshold = 40
    
    cond1 = (abs(tri_c[:,2])< threshold).astype('uint8')
    cond2 = (abs(tri_p[:,2])< threshold).astype('uint8')
    cond = (cond1*cond2).astype('bool')
    
    tri_c = tri_c[cond]
    tri_p = tri_p[cond]
    
    
    # kp1 is imgL_p
    # kp2 is imgL_c
    
    ####Homogeneous Image Coordinate#####
    kp1 = data[cond][:,:2]
    kp2 = data[cond][:,2:]
    
    kp1 = np.concatenate((kp1,np.ones((len(kp1),1))),axis=1)
    kp2 = np.concatenate((kp2,np.ones((len(kp2),1))),axis=1)
    
    """
    Using world coordinates, image co-ordinates and Projection matric calculate Motion matrix T
    """
    ####Projection Matrix######
    P = np.asarray([[7.183351e+02, 0.000000e+00, 6.003891e+02, 4.450382e+01],
                    [0.000000e+00, 7.183351e+02, 1.815122e+02, -5.951107e-01], 
                    [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.616315e-03]])
    

    err = cost_func(T,tri_p, tri_c, kp1, kp2, P)
    #err = err/len(kp1[:,0])
        
    return err
