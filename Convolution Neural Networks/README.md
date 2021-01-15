# MNIST Digit prediction using Vector quantization and CNN
## Files used in this project are
## -Vector Quantization-1600 clusters , 9x9 patches .ipynb
## -Vector Quantization-2500 clusters , simple patches .ipynb  (patch size 9X9)
## -MNIST Prediction using CNN.ipynb (2 CNN layer(maxpool), 1FCN with softmax activation)

- Pre Neural Network era time image classification techniques were used to predict MNIST Data.
- Multiple image patch of size 9*9 were extracted from the image of size 28*28.
- Image patches were clustered into 2500 clusters using a hierarchical clustering method.
- For each image, a histogram based on cluster indices is created.
- This histogram is fed to RandomForest.
- 94% accuracy was achieved while a simple CNN reached 97%.

# Different regualrizartion techniques in CNN and their effect on performance
- Demonstrates the effect of normlaization technique like Batch Norm, Dropout pon convergence and accuracy
- Effect of increasinglayer is also tried on basic model.
- Effect of changing kernal size, Data Augmentation, adding momemntumn term to optimizer is also tried here.
