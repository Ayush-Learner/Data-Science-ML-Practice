#!/usr/bin/env python
# coding: utf-8

# In[14]:


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
from math import cos, sin, tan, pi, log, atan2
import networkx as nx
from networkx import find_cliques
import random
from scipy.spatial.transform import Rotation as R

from keypoints import get_sift_data, get_best_matches
from triangulation_disparity import disparity, triangulation
from ransac import ransac, compute_motion, compute_error
from utils import plot_inlier_matches
from optimization import genEulerZXZMatrix, cost_func


# In[2]:


# os.getcwd()
# os.chdir("D:\\data_odometry_gray\\dataset")


# In[3]:


####Projection Matrix######
P = np.asarray([[7.183351e+02, 0.000000e+00, 6.003891e+02, 4.450382e+01],
                [0.000000e+00, 7.183351e+02, 1.815122e+02, -5.951107e-01], 
                [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.616315e-03]])


# In[4]:


path = os.path.join(os.getcwd(),'data','2011_09_29_drive_0071_sync')
left_path = os.path.join(path,'image_02','data')
right_path = os.path.join(path,'image_03','data')

# path = os.getcwd()
# left_path = os.path.join(path,'sequences\\00\\image_2')
# right_path = os.path.join(path,'sequences\\00\\image_3')

#N = len(os.listdir(left_path))

# imgL = cv2.imread(os.path.join(left_path,"0000000000.png"),0)
# imgR = cv2.imread(os.path.join(right_path,"0000000000.png"),0)


# data = get_best_matches(img1, img2, 200)
# fig, ax = plt.subplots(figsize=(20,10))
# plot_inlier_matches(ax, img1, img2, data)
# fig.savefig('sift_match.pdf', bbox_inches='tight')


# In[5]:


### Hyperparameters #####
loop = 40
Threshold = 2


N = len(os.listdir(left_path))
#N = 300

T = np.zeros(6)
T_list = []

# psi, theta, sigma = 90,90,0
# mat = R.from_euler('xyz', [psi, theta, sigma], degrees=True)
# mat = mat.as_matrix()
# rot = mat
# lin = np.asarray([0, 0, 0]).reshape(3,1)
# pose_update = np.concatenate((rot,lin),axis=1)
# pose = np.vstack((pose_update,np.asarray([0, 0, 0, 1])))
pose = [0,0,0,1]


pose_update_list = []
pose_list = []
cost_list = []
fitting_cost = []
reproj_cost_mean = []
reproj_cost_median = []


for i in range(1,N):
#     i=28
    print("MAIN LOOP",i)
    ####sequence at time stamp T###########
    imgL_c = cv2.imread(os.path.join(left_path,os.listdir(left_path)[i]),0)
    imgR_c = cv2.imread(os.path.join(right_path,os.listdir(right_path)[i]),0)
    
    ####sequence at time stamp T-1###########
    imgL_p = cv2.imread(os.path.join(left_path,os.listdir(left_path)[i-1]),0)
    imgR_p = cv2.imread(os.path.join(right_path,os.listdir(right_path)[i-1]),0)
    
    ###find Match####
    data = get_best_matches(imgL_p, imgL_c, 100)
    
    ###Disparity calculation####
    disp_c = disparity(imgL_c,imgR_c)
    disp_p = disparity(imgL_p,imgR_p)
    
    ###finding 3D coord ###
    tri_c = triangulation(disp_c,data[:,2:])
    tri_p = triangulation(disp_p,data[:,:2])
    
    #finding index of inliers
    th = .2
    dist_c = scipy.spatial.distance.cdist(tri_c, tri_c, 'sqeuclidean')
    dist_p = scipy.spatial.distance.cdist(tri_p, tri_p, 'sqeuclidean')

    mask = (abs(dist_c-dist_p) < th).astype('uint8')
    G = nx.from_numpy_matrix(mask)        
    list_cliq = list(find_cliques(G))
    length = np.asarray([len(i) for i in list_cliq])
    max_cliq_node = list_cliq[np.argmax(length)]
    
    ####inliers####
    world_c_h = tri_c[max_cliq_node]
    world_p_h = tri_p[max_cliq_node]
    
    img_c = data[:,2:][max_cliq_node]
    img_p = data[:,:2][max_cliq_node]
    
    img_p_h = np.concatenate((img_p,np.ones((len(img_p),1))),axis=1)
    img_c_h = np.concatenate((img_c,np.ones((len(img_c),1))),axis=1)
    
    print(img_c_h.shape)
    ########feeding in random idx######
    T = [random.uniform(0, 1) for k in range(0,6)]
    reproj_err_max = 10000
    reproj_err_th = 10
    
    count = 0
    while reproj_err_max>reproj_err_th:
        
        temp_list = []
        ransac_loop = 10
        min_err =1000000000
        
        for j in range(0,ransac_loop):
            #T = np.zeros(6)
            rnd_idx = random.sample(range(0,len(img_p_h)),6)
            U = range(0,len(img_p_h))
            comp_rnd_idx = list(set(U) - set(rnd_idx))
            
            lower = np.asarray([-3.14,-3.14,-3.14,-.5,-.5,-.5])
            upper = np.asarray([3.14,3.14,3.14,.5,.5,.5])

            optRes = least_squares(cost_func, T, method='lm', max_nfev=2000,args=(world_p_h[rnd_idx],
                world_c_h[rnd_idx], img_p_h[rnd_idx], img_c_h[rnd_idx], P))
            T = optRes.x
            cost = optRes.cost/len(img_p_h[:,0])

            err = cost_func(T,world_p_h[comp_rnd_idx], world_c_h[comp_rnd_idx], img_p_h[comp_rnd_idx], img_c_h[comp_rnd_idx], P)
            err = err.reshape((int(len(err)/4),4))
            err = (np.sum(err**2,axis=1)**.5)/2

            temp_list.append(cost)

            if cost<min_err:
                #print("here")
                min_err = cost
                T_update = T
                reproj_err = err
        
        reproj_err_max = max(reproj_err)
        #break
        if reproj_err_max > reproj_err_th:
            count+=1
            idx = comp_rnd_idx[np.argmax(err)]
            world_p_h =  np.concatenate((world_p_h[:idx],world_p_h[idx+1:]),axis=0) 
            world_c_h = np.concatenate((world_c_h[:idx],world_c_h[idx+1:]),axis=0)
            img_p_h = np.concatenate((img_p_h[:idx],img_p_h[idx+1:]),axis=0)
            img_c_h = np.concatenate((img_c_h[:idx],img_c_h[idx+1:]),axis=0)   
            print("Minimizing reprojection error",i)
            print(reproj_err_max)
            
        if count>5: 
            break
            
#     if abs(T_update[0])>.78 or abs(T_update[1])>.78 or abs(T_update[2]) >.78 or abs(T_update[3])>1 or \
#     abs(T_update[4])>1 or abs(T_update[5])>1:
#         continue
            
    if count>5:
         continue
    ##########thresholding cost and pose update#########        
#     ran_th = 3
#     if min_err > ran_th:
#         print("MAIN LOOP",i)
#         print("Error greater than threshold is", min_err)
#         T_update = np.asarray([0,0,0,0,0,0])
#         min_err = ran_th
#         reproj_err = cost_func(T_update,world_p_h[comp_rnd_idx], world_c_h[comp_rnd_idx], img_p_h[comp_rnd_idx], img_c_h[comp_rnd_idx], P)
#         reproj_err = reproj_err.reshape((int(len(reproj_err)/4),4))
#         reproj_err = (np.sum(reproj_err**2,axis=1)**.5)/2
    
    
    
    fitting_cost.append(min_err)
    reproj_cost_mean.append(np.mean(reproj_err))
    reproj_cost_median.append(np.median(reproj_err))
    
    
    ######Pose update###########
    
    rot = genEulerZXZMatrix(T_update[0], T_update[1], T_update[2])
    lin = np.asarray([T_update[3], T_update[4], T_update[5]]).reshape(3,1)
    pose_update = np.concatenate((rot,lin),axis=1)
    pose_update = np.vstack((pose_update,np.asarray([0, 0, 0, 1])))
    
    
    pose_update_list.append(pose_update)
    pose = np.linalg.inv(pose_update).dot(pose)
    T_list.append(T_update)
    pose_list.append(pose)
    cost_list.append(min_err)
#     if i==4:
#         break


# In[7]:


pose_list = np.asarray(pose_list).reshape(1058,4)
pose_list


# In[33]:


#alpha, beta, gamma = 0,1.57,0

def rotation(alpha, beta, gamma):
    Rot_x = np.asarray([[1, 0, 0],
                        [0, cos(alpha), -sin(alpha)],
                        [0, sin(alpha), cos(alpha)]])
    Rot_y = np.asarray([[cos(beta), 0, sin(beta)],
                        [0, 1, 0],
                        [-sin(beta), 0, cos(beta)]])
    Rot_z = np.asarray([[cos(gamma), -sin(gamma), 0],
                        [sin(gamma), cos(gamma), 0],
                        [0, 0, 1]])
    rot = (Rot_x.dot(Rot_y)).dot(Rot_z)
    return rot

def inverse(rot): 
    angle_x = atan2(rot[1,0],rot[0,0])*180/pi
    angle_y = atan2(-rot[2,0],cos(angle_x)*rot[0,0] + sin(angle_x)*rot[1,0])*180/pi
    angle_z = atan2(sin(angle_x)*rot[0,2]-cos(angle_x)*rot[1,2],-sin(angle_x)*rot[0,1]+cos(angle_x)*rot[1,2])*180/pi
    return angle_x, angle_y, angle_z
    
rot = (rotation(0, 0, pi/2).dot(rotation(0, 0, pi/2)))
inverse(rot)


# In[ ]:




