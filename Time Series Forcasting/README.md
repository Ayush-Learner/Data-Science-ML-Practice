# Pollution level forcasting using Bi-Directional LSTM and other Time-Series Covariates 

- Bidirectional LSTM is tried to predict pollution level based on other timeseires like temperature, Humidity, Dew etc
- Multiple time lags were tried but single time lag was giving best result.
- .76 MAE is obtained for single lag while for multiple lag(3 lags) 1.76 MAE is obtained.
- Poor MAE can be attributed to overfitting.
a

## Results from Single Lag
Classifier    |    F1 Score   |   Accuracy    |      AP      |
------------- | ------------- | ------------- | -------------| 
XG Boost  | 87.46  | 82.1  | 0.78 |
Random Forest  | 86.13  | 81.77  | 0.76 |
Gradient Boosting  | 85.8  | 81.73  | 0.79 |
Decision Tree | 85.63  | 80.24  | 0.63 |
SVM  | 84.65  | 79.58  |  |
Logistic Regression  | 79.86  | 77.54  | 0.69 |

## Results from Multiple Lag
![Train Loss](https://github.com/Ayush-Learner/Data-Science-ML-Practice/blob/master/Time%20Series%20Forcasting/Image/Train%20loss%20for%20multiple%20lag.png)
*Train Loss*
![Prediction](https://github.com/Ayush-Learner/Data-Science-ML-Practice/blob/master/Time%20Series%20Forcasting/Image/prediction%20for%20multiple%20lag.png)
*Prediction*
