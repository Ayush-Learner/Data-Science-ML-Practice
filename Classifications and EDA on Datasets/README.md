# Adult dataset analysis and prediction
## About Data Set
Dataset can be found [here](http://archive.ics.uci.edu/ml/datasets/Adult). It is a multivariate problem with task of binary classification. Independent variable are both continuous and categorical.A total of ~49K instances compose this dataset. 
The task is to predict income flag (<=50k,>50k) based on features like gender, race, marital status, education and more.


## Description
- Exhaustive EDA is done. Both univariate and multivariate. Outliers were removed and missing data was imputed by its mean.
- Hypothesis were generated and new features were engineered based on hypothesis.
- Random Forest, SVM, Decision Tree, XGBOOST, Logistic regression were trained. Their Hyper parameter were tuned using grid search.
- Hyper parameter were optimized for F1 metric not accuracy metric.
- As data set was highly imbalnced, different test and traininh ratio were tried in combination with under sampling and over sampling.
- XGBOOST with F1 score of 87% and accuracy of 81% was chosen as final model.

## Best Results by classifiers
Classifier    |    F1 Score   |   Accuracy    |      AP      |
------------- | ------------- | ------------- | -------------| 
Content Cell  | Content Cell  | Content Cell  | Content Cell |
Content Cell  | Content Cell  | Content Cell  | Content Cell |


# PIMA india dataset analysis and prediction
## About Data Set
Dataset can be found [here](https://www.kaggle.com/uciml/pima-indians-diabetes-database). It is a multivariate problem with binary classification task to predict whether Diabetic or not.Dataset contains some 900-100 datapoints with continuous independent variables like Blood Pressure, Age and Categorical are like # of Pregnency.

## Description
- Cleaned, Visualized and analyzed PIMA Indian dataset.
- Performed bivariate and univariate analyses on features and generated features based on the analysis.
- Further features were generated based on the hypothesis.
- Models were optimized on F1 score. F1 score of 81% and accuracy 74% is obtained by Random Forest.
- Attributes affecting Diabetes were predicted through Decision Tree-based approach.
