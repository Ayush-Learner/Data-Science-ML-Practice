# Pollution level forcasting using Bi-Directional LSTM and other Time-Series Covariates 

- Bidirectional LSTM is tried to predict pollution level based on other timeseires like temperature, Humidity, Dew etc
- Multiple time lags were tried but single time lag was giving best result.
- .76 MAE is obtained for single lag while for multiple lag(3 lags) 1.76 MAE is obtained.
- Poor MAE can be attributed to overfitting.
a

## Results from Single Lag
![Train Loss](https://github.com/Ayush-Learner/Data-Science-ML-Practice/blob/master/Time%20Series%20Forcasting/Image/Train%20loss%20for%20single%20lag.png)

*Train Loss*
![Prediction](https://github.com/Ayush-Learner/Data-Science-ML-Practice/blob/master/Time%20Series%20Forcasting/Image/prediction%20for%20single%20lag.png)

*Prediction*

## Results from Multiple Lag
![Train Loss](https://github.com/Ayush-Learner/Data-Science-ML-Practice/blob/master/Time%20Series%20Forcasting/Image/Train%20loss%20for%20multiple%20lag.png)

*Train Loss*
![Prediction](https://github.com/Ayush-Learner/Data-Science-ML-Practice/blob/master/Time%20Series%20Forcasting/Image/prediction%20for%20multiple%20lag.png)

*Prediction*


## Summary
Lag    |    Training Loss   |   Prediction MAE    |
------------- | ------------- | ------------- | 
Single Lag  | 0.069  | 0.76  |
Multiple Lag  | 0.067  | 1.70  |
