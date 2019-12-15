# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:46:52 2019

@author: melekel
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("D:\\Users\\melekel\\Desktop\\Melek\\Master_BA\\Programlama_Masoud\\Proje\\HR_Data.csv")
data.head()

# Na control
data.isnull().sum()


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
# Correalation
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


# Grid search CV for Logistic Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

parameters = {"C":[20.0, 40.0, 60.0, 80.0, 100.0, 120.0], "penalty":["l1","l2"]}# l1 lasso l2 ridge
logr = LogisticRegression()
logreg_gridcv = GridSearchCV(logr, parameters, cv=10)
logreg_gridcv.fit(x_train,y_train)

print("Best Parameters for Logistic Regression are :",logreg_gridcv.best_params_)
print("accuracy :",logreg_gridcv.best_score_)

best_params_logreg = logreg_gridcv.best_params_
logreg2 = LogisticRegression(C = 20.0,
                             penalty = "l1")
logreg2.fit(x_train, y_train)
y_pred = logreg2.predict(x_test)

from sklearn.metrics import accuracy_score
acc_logreg2 = accuracy_score(y_pred, y_test)
print("Logistic Regression Accuracy Score with Grid Search CV is :", acc_logreg2)

# Export model
import pickle 
with open('LogReg.pkl', 'wb') as model_file:
  pickle.dump(logreg2, model_file)


# Grid search CV for Adaboost
from sklearn.ensemble import AdaBoostClassifier

parameters = {"base_estimator" : ["gini", "entropy"],
              "learning_rate" : [0.01,0.05,0.1,0.3,1],
              "n_estimators" : [50],
              "algorithm" : ['SAMME']
              }
ada = AdaBoostClassifier()
ada_gridcv = GridSearchCV(ada, parameters, scoring = 'roc_auc')
ada_gridcv.fit(x_train,y_train)

print("Best Parameters for Adaboost are :",ada_gridcv.best_params_)
print("accuracy :",ada_gridcv.best_score_)

best_params_ada = ada_gridcv.best_params_
ada2 = AdaBoostClassifier(C = 20.0,
                             penalty = "l1")
ada2.fit(x_train, y_train)
y_pred = ada2.predict(x_test)

from sklearn.metrics import accuracy_score
acc_ada2 = accuracy_score(y_pred, y_test)
print("Adaboost Accuracy Score with Grid Search CV is :", acc_ada2)
 