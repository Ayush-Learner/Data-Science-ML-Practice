<!-- This Project is adapted from [here](https://github.com/qinnzou/Robust-Lane-Detection). -->

# Project Description
Target of this project is to detect road lanes using sequence of image data from car cam. Sequence of image is fed in to network and binary pixel image is obtained as output. pizel with value 0 is background and rest is lane.
Dataset used is [here](https://drive.google.com/drive/folders/1rpPgQ9TmG99eQ22Rh5Of8x8c0iOjyq4Y?usp=sharing). It is TuSimple dataset.

# Approach
- Sequence of images are fed in to network.
- genertaed density output is passed through Hough Transform to generate lines.
- Possible lines are filtered out and nearby lines are clustered (using K-Means) to obtain smooth output. 


![Input](https://github.com/Ayush-Learner/Data-Science-ML-Practice/blob/master/Road%20Lane%20segmentation/Images/input_alt_10.PNG)
![Prediction](https://github.com/Ayush-Learner/Data-Science-ML-Practice/blob/master/Road%20Lane%20segmentation/Images/prediction_alt_10.PNG)
![Line detection](https://github.com/Ayush-Learner/Data-Science-ML-Practice/blob/master/Road%20Lane%20segmentation/Images/Multiple_hough_lines.jpg)
![Final output](https://github.com/Ayush-Learner/Data-Science-ML-Practice/blob/master/Road%20Lane%20segmentation/Images/mean_hough_lines_alt_10.jpg)
![Ground truth](https://github.com/Ayush-Learner/Data-Science-ML-Practice/blob/master/Road%20Lane%20segmentation/Images/gt_alt_10.PNG)

# Network Architecture and training
- Encoder Decoder based architecture is adopted with VGG16 weights trained on imagenet dataset.
- To hold temporal data 2 Layers Conv-LSTM units were used.
- As data was highly unbalanced, weighted cross-entropy loss was used with adam optimizer.
- Different training strategies involving # of sequence, Loss function, Optimzer, strides in sequences and even different weights were also tried.
- The best performance was obtained for weight [1,.4], Adam optimzer, 5 numbers of sequence with stride of 1.
- Pixel wise accuracy of 98% is obtained but still lagging in F1 metrics with 32%.




# Results

Results are shown without post processing.

## Good Results
![98%](https://github.com/Ayush-Learner/Data-Science-ML-Practice/blob/master/Road%20Lane%20segmentation/Images/98%25%20accuracy.png)
## Bad results
![98%](https://github.com/Ayush-Learner/Data-Science-ML-Practice/blob/master/Road%20Lane%20segmentation/Images/download.png)

# TODO
- Increase F1 score.
- Reduce # of parameters to increase fps.
- Real life implementation using kalman filter.
