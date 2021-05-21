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
from scipy.spatial.transform import Rotation as R

# def genEulerZXZMatrix(theta, phi, psi):
    

#     theta = 0
#     phi = 0
    
#     mat = np.zeros((3,3))
#     mat[0,0] = cos(phi)*cos(psi)
#     mat[0,1] = sin(theta)*sin(phi)*cos(psi) - cos(theta)*sin(psi)
#     mat[0,2] = cos(theta)*sin(phi)*cos(psi) + sin(theta)*sin(psi)
    
#     mat[1,0] = cos(phi)*sin(psi)
#     mat[1,1] = sin(theta)*sin(phi)*sin(psi) + cos(theta)*cos(psi)
#     mat[1,2] = cos(theta)*sin(phi)*sin(psi) - sin(theta)*cos(psi)
    
#     mat[2,0] = -sin(phi)
#     mat[2,1] = sin(theta)*cos(phi)
#     mat[2,2] = cos(theta)*cos(phi)
# #     mat = np.eye(3)
    
#     return mat



def genEulerZXZMatrix(psi, theta, sigma):
    # ref http://www.u.arizona.edu/~pen/ame553/Notes/Lesson%2008-A.pdf 
    
#     theta  = 0
#     sigma = 0
#     mat = np.zeros((3,3))
#     mat[0,0] = cos(psi) * cos(sigma) - sin(psi) * cos(theta) * sin(sigma)
#     mat[0,1] = -cos(psi) * sin(sigma) - sin(psi) * cos(theta) * cos(sigma)
#     mat[0,2] = sin(psi) * sin(theta)
    
#     mat[1,0] = sin(psi) * cos(sigma) + cos(psi) * cos(theta) * sin(sigma)
#     mat[1,1] = -sin(psi) * sin(sigma) + cos(psi) * cos(theta) * cos(sigma)
#     mat[1,2] = -cos(psi) * sin(theta)
    
#     mat[2,0] = sin(theta) * sin(sigma)
#     mat[2,1] = sin(theta) * cos(sigma)
#     mat[2,2] = cos(theta)

    mat = R.from_euler('zxz', [psi, theta, sigma], degrees=False)
    mat = mat.as_matrix()
    
    return mat

# def cost_func(pose_vec, w_p, w_c, i_p, i_c, P):
    
#     #print(P)
    
#     rot = genEulerZXZMatrix(pose_vec[0], pose_vec[1], pose_vec[2])
#     lin = np.asarray([pose_vec[3], pose_vec[4], pose_vec[5]]).reshape(3,1)
#     pose = np.concatenate((rot,lin),axis=1)
#     pose = np.vstack((pose,np.asarray([0, 0, 0, 1])))
    
#     ###Error calculation###
    
#     j_pred_p = (P.dot(pose)).dot(w_c.T).T
#     j_pred_p = j_pred_p/np.repeat(j_pred_p[:,-1].reshape(len(w_c),1),3,axis=1)
#     err_p = i_p[:,:2] - j_pred_p[:,:2] 
    
#     j_pred_c = (P.dot(np.linalg.inv(pose))).dot(w_p.T).T
#     j_pred_c = j_pred_c/np.repeat(j_pred_c[:,-1].reshape(len(w_c),1),3,axis=1)
#     err_c = i_c[:,:2] - j_pred_c[:,:2]
                
#     tot_err = np.concatenate((err_p, err_c),axis=1)  
#     #print(tot_err.shape)
#     #print(tot_err)           
        
#     return tot_err.flatten()       



def cost_func(pose_vec, w_p, w_c, i_p, i_c, P):
    
    #print(P)
    
    #rot = cv2.Rodrigues(np.asarray([pose_vec[0], pose_vec[1], pose_vec[2]]))[0]
    rot = genEulerZXZMatrix(pose_vec[0], pose_vec[1], pose_vec[2])
    lin = np.asarray([pose_vec[3], pose_vec[4], pose_vec[5]]).reshape(3,1)
    pose = np.concatenate((rot,lin),axis=1)
    pose = np.vstack((pose,np.asarray([0, 0, 0, 1])))
    
    ###Error calculation###
    
    j_pred_p = (P.dot(pose)).dot(w_c.T).T
    #j_pred_p = (P.dot(np.linalg.inv(pose))).dot(w_c.T).T
    j_pred_p = j_pred_p/np.repeat(j_pred_p[:,-1].reshape(len(w_c),1),3,axis=1)
    err_p = i_p[:,:2] - j_pred_p[:,:2] 
    
    j_pred_c = (P.dot(np.linalg.inv(pose))).dot(w_p.T).T
    j_pred_c = j_pred_c/np.repeat(j_pred_c[:,-1].reshape(len(w_c),1),3,axis=1)
    err_c = i_c[:,:2] - j_pred_c[:,:2]
                
    tot_err = np.concatenate((err_p, err_c),axis=1)  
    #print(tot_err.shape)
    #print(tot_err)           
        
    return tot_err.flatten()     
