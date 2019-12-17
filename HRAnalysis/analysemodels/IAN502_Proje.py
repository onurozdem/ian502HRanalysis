# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:46:52 2019

@author: melekel
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

data = pd.read_csv("D:\\Users\\melekel\\Desktop\\Melek\\Master_BA\\Programlama_Masoud\\Proje\\HR_Data.csv")
data.head()

# Na control
data.isnull().sum()

# These columns have same value on rows. Therefore they are removed.
data = data.drop(['EmployeeCount','EmployeeNumber','Over18'], axis = 1)

# Label Encoder
"""
Attrition,
BusinessTravel,
Department,
EducationField,
Gender,
MaritalStatus,
Over18,
OverTime
"""
le = LabelEncoder()
data2 = data.apply(le.fit_transform)

"""
# =============================================================================
# Correalation
# =============================================================================
corr = data2.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(40, 220, n=300, center = "light"),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
"""
# Spliting target column
target_attrition = data2['Attrition']
data3 = data2.drop(['Attrition'], axis = 1)

# Modeling
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data3, 
                                                    target_attrition, 
                                                    test_size = 0.2, 
                                                    random_state = 0)


# =============================================================================
# Grid search CV for Logistic Regression
# =============================================================================
from sklearn.linear_model import LogisticRegression

parameters = {"C":[20.0, 40.0, 60.0, 80.0, 100.0, 120.0], "penalty":["l1","l2"]}# l1 lasso l2 ridge
logr = LogisticRegression()
gridcv_logreg = GridSearchCV(logr, parameters, cv=10)
gridcv_logreg.fit(x_train,y_train)

print("Best Parameters for Logistic Regression are :", gridcv_logreg.best_params_)
print("accuracy :",gridcv_logreg.best_score_)

best_params_logreg = gridcv_logreg.best_params_
logreg2 = LogisticRegression(C = 20.0,
                             penalty = "l1")
logreg2.fit(x_train, y_train)
y_pred = logreg2.predict(x_test)

acc_logreg2 = accuracy_score(y_pred, y_test)
print("Logistic Regression Accuracy Score with Grid Search CV is :", acc_logreg2)

"""
# Export model
import pickle 
with open('LogReg.pkl', 'wb') as model_file:
  pickle.dump(logreg2, model_file)
"""
# =============================================================================
# Grid search CV for Adaboost
# =============================================================================
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()
parameters = {
              "learning_rate" : [0.01, 0.05, 0.1, 0.3, 1, 2],
              "n_estimators" : [50, 100, 1000],
              "algorithm" : ["SAMME", "SAMME.R"]
              }
gridcv_ada = GridSearchCV(estimator = ada, 
                          param_grid = parameters, 
                          scoring = 'accuracy',
                          cv = 10)
gridcv_ada.fit(x_train,y_train)

print("Best Parameters for Adaboost are :",gridcv_ada.best_params_) 
# Best Parameters for Adaboost are : {'algorithm': 'SAMME', 'learning_rate': 0.3, 'n_estimators': 1000}
print("accuracy :",gridcv_ada.best_score_) # 0.8852040816326531

best_params_ada = gridcv_ada.best_params_

ada2 = AdaBoostClassifier(algorithm = "SAMME",
                          learning_rate = 0.3,
                          n_estimators = 1000)
ada2.fit(x_train, y_train)
y_pred = ada2.predict(x_test)

acc_ada2 = accuracy_score(y_pred, y_test) # 0.8707482993197279
print("Adaboost Accuracy Score with Grid Search CV is :", acc_ada2)

# =============================================================================
# Grid search CV for Naive Bayes
# =============================================================================
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)
acc_nb = accuracy_score(y_pred, y_test)
print("Naive Bayes Accuracy Score is :", acc_nb) # 0.8095238095238095

# =============================================================================
# Grid search CV for Decision Tree
# =============================================================================

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
parameters = {"criterion" : ["gini","entropy"],
              "max_depth" : np.arange(1,26,1)}

gridcv_dt = GridSearchCV(estimator = dt,
                         param_grid = parameters,
                         scoring = "accuracy",
                         cv = 10)

gridcv_dt.fit(x_train, y_train)

print("Best Parameters for Decision Tree are :",gridcv_dt.best_params_) 
# Best Parameters for Decision Tree are : {'criterion': 'entropy', 'max_depth': 4}
print("accuracy :",gridcv_dt.best_score_) # 0.8418367346938775

best_params_dt = gridcv_dt.best_params_

dt2 = DecisionTreeClassifier(criterion = "entropy", max_depth = 4)

dt2.fit(x_train, y_train)

y_pred = dt2.predict(x_test)

acc_dt2 = accuracy_score(y_pred, y_test)
print("Decision Tree Accuracy Score with Grid Search CV is :", acc_dt2)





