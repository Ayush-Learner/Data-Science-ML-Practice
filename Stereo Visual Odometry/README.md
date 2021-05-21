# Stereo Visual odometry using Robust inlier selection method

# Introduction
In this project visual odometry is implmented using stereo vision. Stereo Vision data is obtained from KITTI Dataset and can be obtained from [here](http://www.cvlibs.net/datasets/kitti/raw_data.php). In this sequence GPS data is taken as ground truth. This implementtaion is inspired from famous NASA paper and can be accessed from [here](https://www-robotics.jpl.nasa.gov/publications/Andrew_Howard/howard_iros08_visodom.pdf).

# Steps
-- ORB feature were extracted and disparity was calculated on already rectified images using OpenCV implmentation.
-- Each extracted feature's 3D coordinate is triangulated.
-- Correspondence between features across frame is selected by minimum eucledian distance for each other.
-- Robust features are selected using 3D world constraints, which is distance between 3D coordinate of pair of features will not change in world.
-- All such consistent pairs are selected by formulating problem as maximum size clique and solved using NetworkX implmenetation.
-- For all such features and their 3D coordinates, reprojection error is minimzed using minimal set of features.
-- Motion matrix corresponding to minimum reprojection error for out of set features is considered final motion matrix.
-- A version of RANSAC was tried to further eliminate feature was tried on above step, to bring reprojection error below threshold but it was making computation less and less real time.
-- Obtained motion matrix is compunded for each sequence to obtain final pose.
# Results
In reults we show results obtained for each frame-by-frame computation.
## Mean & Median Reprojection error and Fitting Cost

| Mean reprojection error | Median reprojection error | Fitiing Cost |
:-----:|:------:|:------:
![](https://github.com/Ayush-Learner/Data-Science-ML-Practice/blob/master/Stereo%20Visual%20Odometry/Images/mean%20reprojection%20error.png)|![](https://github.com/Ayush-Learner/Data-Science-ML-Practice/blob/master/Stereo%20Visual%20Odometry/Images/median%20reprojection%20error.png)|![](https://github.com/Ayush-Learner/Data-Science-ML-Practice/blob/master/Stereo%20Visual%20Odometry/Images/fittinng%20cost.png)

## Forward motion
## Sideways motion
## Steering motion
