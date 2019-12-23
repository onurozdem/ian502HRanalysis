# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:46:52 2019

@author: melekel
"""

import datetime
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
# Feature Selection
# =============================================================================



# =============================================================================
# Grid search CV for Logistic Regression
# =============================================================================
from sklearn.linear_model import LogisticRegression

parameters = {"C":[20.0, 40.0, 60.0, 80.0, 100.0, 120.0], "penalty":["l1","l2"]}# l1 lasso l2 ridge
logr = LogisticRegression()
gridcv_logreg = GridSearchCV(logr, parameters, cv=10)

print("Grid Search started for Logistic Regression: ", datetime.datetime.now())
gridcv_logreg.fit(x_train,y_train)
print("Grid Search finished for Logistic Regression: ", datetime.datetime.now())

print("Best Parameters for Logistic Regression are :", gridcv_logreg.best_params_)
print("accuracy :",gridcv_logreg.best_score_)

logreg2 = LogisticRegression(C = 20.0,
                             penalty = "l1")
logreg2.fit(x_train, y_train)
y_pred = logreg2.predict(x_test)
acc_logreg2 = accuracy_score(y_pred, y_test)
print("Logistic Regression Accuracy Score with Grid Search CV is : ", acc_logreg2)

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

print("Grid Search started for Adaboost: ", datetime.datetime.now())
gridcv_ada.fit(x_train,y_train)
print("Grid Search finished for Adaboost: ", datetime.datetime.now())

print("Best Parameters for Adaboost are :",gridcv_ada.best_params_)
print("accuracy :",gridcv_ada.best_score_)

ada2 = AdaBoostClassifier(algorithm = "SAMME",
                          learning_rate = 0.3,
                          n_estimators = 1000)
ada2.fit(x_train, y_train)
y_pred = ada2.predict(x_test)
acc_ada2 = accuracy_score(y_pred, y_test)
print("Adaboost Accuracy Score with Grid Search CV is : ", acc_ada2)
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

print("Grid Search started for Decision Tree: ", datetime.datetime.now())
gridcv_dt.fit(x_train, y_train)
print("Grid Search finished for Decision Tree: ", datetime.datetime.now())

print("Best Parameters for Decision Tree are :",gridcv_dt.best_params_)
print("accuracy :",gridcv_dt.best_score_)

dt2 = DecisionTreeClassifier(criterion = "entropy", max_depth = 4)
dt2.fit(x_train, y_train)
y_pred = dt2.predict(x_test)
acc_dt2 = accuracy_score(y_pred, y_test)
print("Decision Tree Accuracy Score with Grid Search CV is : ", acc_dt2)
# =============================================================================
# XGBoost -- APPLY PERSONAL PC
# =============================================================================
#import xgboost as xgb



# =============================================================================
# Linear Discriminant Analysis
# =============================================================================

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(shrinkage = "auto",
                                 solver = "lsqr", # eigen, svd(default)
                                 )
lda.fit(x_train, y_train)
y_pred = lda.predict(x_test)
acc_lda = accuracy_score(y_pred, y_test)
print("Linear Discriminant Analysis Accuracy Score is : ", acc_lda)

# =============================================================================
# Grid search CV for Naive Bayes
# =============================================================================
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)
acc_nb = accuracy_score(y_pred, y_test)
print("Naive Bayes Accuracy Score is : ", acc_nb) # 0.8095238095238095

# =============================================================================
# KNN
# =============================================================================

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

parameters = {"leaf_size" : np.arange(2,20,1)}
gridcv_knn = GridSearchCV(estimator = knn,
                          param_grid = parameters,
                          scoring = "accuracy",
                          cv = 10)

print("Grid Search started for KNN : ", datetime.datetime.now())
gridcv_knn.fit(x_train, y_train)
print("Grid Search finished for KNN : ", datetime.datetime.now())

print("Best Parameters for KNN are :",gridcv_knn.best_params_)
print("accuracy :",gridcv_knn.best_score_)

knn2 = KNeighborsClassifier(leaf_size = 2)
knn2.fit(x_train, y_train)
y_pred = knn2.predict(x_test)
acc_knn2 = accuracy_score(y_pred, y_test)
print("KNN Accuracy Score is :", acc_knn2)

# =============================================================================
# Random Forest
# =============================================================================

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
"""
parameters = {"n_estimators" : np.arange(100, 500, 100),
              "max_features": ["log2", "sqrt", "auto"],
              "max_depth": np.arange(2, 10, 1),
              "criterion" : ["gini", "entropy"]}
"""
parameters = {"n_estimators" : np.arange(100, 500, 100),
              "max_features": ["auto"],
              "max_depth": np.arange(2, 10, 1),
              "criterion" : ["gini", "entropy"]}
gridcv_rf = GridSearchCV(estimator = rf,
                          param_grid = parameters,
                          scoring = "accuracy",
                          cv = 10)

print("Grid Search started for Random Forest: ", datetime.datetime.now())
gridcv_rf.fit(x_train, y_train)
print("Grid Search finished for Random Forest: ", datetime.datetime.now())

print("Best Parameters for Random Forest are :",gridcv_rf.best_params_)
print("accuracy :",gridcv_rf.best_score_)

rf2 = RandomForestClassifier(criterion = "entropy",
                             max_depth = 8,
                             max_features = "auto",
                             n_estimators = 100)
rf2.fit(x_train, y_train)
y_pred = rf2.predict(x_test)
acc_rf2 = accuracy_score(y_pred, y_test)
print("Random Forest Accuracy Score with Grid Search CV is : ", acc_rf2)

# =============================================================================
# Neural Network
# =============================================================================


# =============================================================================
# SVM
# =============================================================================
from sklearn.svm import SVC

svm = SVC()
"""
# 1.5 saat calisti sonuc vermedi.
parameters = {"C" : np.arange(100, 1000, 100),
              "gamma" : [0.01, 0.001, 0.0001],
              "kernel" : ["rbf", "linear"]}

# 30 dk calisti sonuc vermedi.
parameters = {"C" : [100,1000],
              "gamma" : [0.001, 0.0001],
              "kernel" : ["rbf", "linear"]}
"""
parameters = {"C" : np.arange(100, 1000, 100),
              "gamma" : [0.01, 0.001, 0.0001],
              "kernel" : ["rbf"]}


gridcv_svm = GridSearchCV(estimator = svm,
                          param_grid = parameters,
                          scoring = "accuracy",
                          cv = 10)

print("Grid Search started for SVM: ", datetime.datetime.now())
gridcv_svm.fit(x_train, y_train)
print("Grid Search finished for SVM: ", datetime.datetime.now())

print("Best Parameters for SVM are :",gridcv_svm.best_params_)
print("accuracy :",gridcv_svm.best_score_)

svm2 = SVC(C = 100,
           gamma = 0.001,
           kernel = "rbf",
           probability = True)
svm2.fit(x_train, y_train)
y_pred = svm2.predict(x_test)
acc_svm2 = accuracy_score(y_pred, y_test)
print("SVM Score with Grid Search CV is :", acc_svm2)

# =============================================================================
# ROC Curve
# =============================================================================
from sklearn import metrics
lw = 3
# y-axis true-positive, x-axis false-positive
predict_proba_logreg = logreg2.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  predict_proba_logreg)
auc_logreg = metrics.roc_auc_score(y_test, predict_proba_logreg)
plt.plot(fpr,tpr,label="Logistic Regression, auc_logreg = " + str(auc_logreg))
plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='dashed')
plt.legend(loc=4)
plt.show()

pred_proba_ada = ada2.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  pred_proba_ada)
auc_ada = metrics.roc_auc_score(y_test, pred_proba_ada)
plt.plot(fpr,tpr,label="Adaboost, auc_ada = " + str(auc_ada))
plt.legend(loc=4)
plt.show()

pred_proba_dt = dt2.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  pred_proba_dt)
auc_dt = metrics.roc_auc_score(y_test, pred_proba_dt)
plt.plot(fpr,tpr,label="Decision Tree, auc_dt = " + str(auc_dt))
plt.legend(loc=4)
plt.show()

pred_proba_lda = lda.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  pred_proba_lda)
auc_lda = metrics.roc_auc_score(y_test, pred_proba_lda)
plt.plot(fpr,tpr,label="Linear Discriminant Analysis, auc_lda = " + str(auc_lda))
plt.legend(loc=4)
plt.show()

pred_proba_nb = nb.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  pred_proba_nb)
auc_nb = metrics.roc_auc_score(y_test, pred_proba_nb)
plt.plot(fpr,tpr,label="Naive Bayes, auc_nb = " + str(auc_nb))
plt.legend(loc=4)
plt.show()

pred_proba_knn = knn2.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  pred_proba_knn)
auc_knn = metrics.roc_auc_score(y_test, pred_proba_knn)
plt.plot(fpr,tpr,label="KNN, auc_knn = " + str(auc_knn))
plt.legend(loc=4)
plt.show()

pred_proba_rf = rf2.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  pred_proba_rf)
auc_rf = metrics.roc_auc_score(y_test, pred_proba_rf)
plt.plot(fpr,tpr,label="Random Forest, auc_rf = " + str(auc_rf))
plt.legend(loc=4)
plt.show()

pred_proba_svm = svm2.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  pred_proba_svm)
auc_svm = metrics.roc_auc_score(y_test, pred_proba_svm)
plt.plot(fpr,tpr,label="Support Vector Machine, auc_svm = " + str(auc_svm))
plt.legend(loc=4)
plt.show()




