# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:46:52 2019

@author: melekel
"""
# =============================================================================
# Libraries
# =============================================================================
import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics

# =============================================================================
# Import Data
# =============================================================================
data = pd.read_csv("D:\\Users\\melekel\\Desktop\\Melek\\Master_BA\\Programlama_Masoud\\Proje\\HR_Data.csv")
data.head()

# =============================================================================
# Na control
# =============================================================================
data.isnull().sum()

# =============================================================================
# Visulisation
# =============================================================================

# Age Distribution
sns.distplot(data['Age'])
plt.show()

sns.boxplot(data['Gender'], data['Age'])
sns.boxplot(data['Gender'], data['MonthlyIncome'])
sns.boxplot(data['JobRole'], data['MonthlyIncome'])
sns.boxplot(data['DistanceFromHome'], data['MonthlyIncome'])

# =============================================================================
# Label Encoder
# =============================================================================
"""
Attrition,BusinessTravel,Department,EducationField,Gender,MaritalStatus,OverTime
"""
le = LabelEncoder()
data2 = data.apply(le.fit_transform)

"""
# =============================================================================
# Corrlation
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

# =============================================================================
# Split and drop target column 
# =============================================================================

target_attrition = data2['Attrition']
data3 = data2.drop(['Attrition'], axis = 1)

# Drop unnecessary columns
data3 = data3.drop(['EmployeeCount'], axis = 1)

# =============================================================================
# Feature Selection
# =============================================================================

rf = RandomForestClassifier()
rf.fit(data3,target_attrition)

feat_imp = pd.DataFrame()
feat_imp["FeatName"] = data3.columns
feat_imp["FeatImportance"] = pd.DataFrame(rf.feature_importances_, columns = ["FeatImportance"])

importance_thres = 0.02
data4 = pd.DataFrame()

for i in range(0, len(feat_imp)):
    if feat_imp["FeatImportance"][i] > importance_thres:
        column = feat_imp["FeatName"][i]
        data4[column] = data3[column]
        
        
# =============================================================================
# Standardize Numeric Columns
# =============================================================================
# Numeric Columns
NumericCols = ['DailyRate', 'DistanceFromHome','HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'TotalWorkingYears',
'YearsAtCompany' , 'YearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager']

#'DailyRate', 'DistanceFromHome' 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'TotalWorkingYears',
#'YearsAtCompany' , 'YearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager'

StSc = StandardScaler()
for i in range(0,len(data4.columns)):
    if data4.columns[i] in NumericCols:
        print(data4.columns[i])
        data4[data4.columns[i]] = pd.DataFrame(StSc.fit_transform(data4.iloc[:,i:(i+1)]))

# =============================================================================
# Split dataset as train and test data
# =============================================================================

x_train, x_test, y_train, y_test = train_test_split(data4, 
                                                    target_attrition, 
                                                    test_size = 0.2, 
                                                    random_state = 0)



# =============================================================================
# Grid search CV for Logistic Regression
# =============================================================================

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

# Export model
with open('LogReg.pkl', 'wb') as model_file:
  pickle.dump(logreg2, model_file)

# =============================================================================
# Grid search CV for Adaboost
# =============================================================================

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

# Export model
with open('AdaBoost.pkl', 'wb') as model_file:
  pickle.dump(ada2, model_file)
# =============================================================================
# Grid search CV for Decision Tree
# =============================================================================

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

# Export model
with open('DTree.pkl', 'wb') as model_file:
  pickle.dump(dt2, model_file)
# =============================================================================
# XGBoost -- APPLY PERSONAL PC
# =============================================================================
#import xgboost as xgb



# =============================================================================
# Linear Discriminant Analysis
# =============================================================================

lda = LinearDiscriminantAnalysis(shrinkage = "auto",
                                 solver = "lsqr", # eigen, svd(default)
                                 )
lda.fit(x_train, y_train)
y_pred = lda.predict(x_test)
acc_lda = accuracy_score(y_pred, y_test)
print("Linear Discriminant Analysis Accuracy Score is : ", acc_lda)

# Export model
with open('LDA.pkl', 'wb') as model_file:
  pickle.dump(lda, model_file)
# =============================================================================
# Grid search CV for Naive Bayes
# =============================================================================
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)
acc_nb = accuracy_score(y_pred, y_test)
print("Naive Bayes Accuracy Score is : ", acc_nb) # 0.8095238095238095

# Export model
with open('NBayes.pkl', 'wb') as model_file:
  pickle.dump(nb, model_file)
# =============================================================================
# KNN
# =============================================================================

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

# Export model
with open('KNN.pkl', 'wb') as model_file:
  pickle.dump(knn2, model_file)
# =============================================================================
# Random Forest 
# =============================================================================

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

rf2.feature_importances_

# Export model
with open('RForest.pkl', 'wb') as model_file:
  pickle.dump(rf2, model_file)
# =============================================================================
# Neural Network
# =============================================================================
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical


classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 31))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN | means applying SGD on the whole ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, epochs=50)

score, acc_annTrain = classifier.evaluate(x_train, y_train,batch_size=10)
print('Train score:', score)
print('Train accuracy:', acc_annTrain)
# Part 3 - Making predictions and evaluating the model
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

print('*'*20)
score, acc_annTest = classifier.evaluate(x_test, y_test,
                            batch_size=10)
print('Test score:', score)
print('Test accuracy:', acc_annTest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

p = sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# =============================================================================
# SVM
# =============================================================================

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

# Export model
with open('SVM.pkl', 'wb') as model_file:
  pickle.dump(svm2, model_file)
# =============================================================================
# ROC Curve
# =============================================================================
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




