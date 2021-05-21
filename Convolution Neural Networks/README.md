# MNIST Digit prediction using Vector quantization and CNN
## Files used in this project are
## -Vector Quantization-1600 clusters , 9x9 patches .ipynb
## -Vector Quantization-2500 clusters , simple patches .ipynb  (patch size 9X9)
## -MNIST Prediction using CNN.ipynb (2 CNN layer(maxpool), 1FCN with softmax activation)

- Similar to Neural net but a computationally less expensive approach to classify MNIST data is tried and its performance is compared with that of CNN.
- Multiple image patch of size 9X9 were extracted from the image of size 28X28.
- Image patches were clustered into 2500 clusters using a hierarchical clustering method.
- For each image, a histogram based on cluster indices is created.
- This step is similar to Final connected layer of CNN which has thousands of parameters.
- This histogram is fed to RandomForest and Naive Bayes.
- For CNN above step has softmax as classifier.
- 94% accuracy was achieved with Naive Bayes while a simple but computationally complex CNN reached 97%.

## Summary 
Classifier    |    Accuracy   |
------------- | ------------- | 
CNN  | 97%  |
1600 Clusters Random Forest  | 94.17  |
1600 Clusters Naive Bayes  | 92.93  |
2500 Clusters Random Forest| 83.67  |
2500 Clusters Naive Bayes | 94.49  |

<!-- # different regualrizartion techniques in CNN and their effect on performance
- Demonstrates the effect of normlaization technique like Batch Norm, Dropout on convergence and accuracy
- Effect of increasing layer is also tried on basic model.
- Effect of changing kernal size, Data Augmentation, adding momemntumn term to optimizer is also tried here.
 -->
