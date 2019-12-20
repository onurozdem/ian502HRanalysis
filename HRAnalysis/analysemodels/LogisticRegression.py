import pickle
import datetime

from sklearn import metrics
from HRAnalysis.models import ModelDetail
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


class LR:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        try:
            model_score_dict = dict()

            """parameters = {"C":[20.0, 40.0, 60.0, 80.0, 100.0, 120.0], "penalty":["l1","l2"]}# l1 lasso l2 ridge
            logr = LogisticRegression()
            gridcv_logreg = GridSearchCV(logr, parameters, cv=10)

            print("Grid Search started for Logistic Regression: ", datetime.datetime.now())
            gridcv_logreg.fit(self.x_train, self.y_train)
            print("Grid Search finished for Logistic Regression: ", datetime.datetime.now())

            print("Best Parameters for Logistic Regression are :", gridcv_logreg.best_params_)
            print("accuracy :",gridcv_logreg.best_score_)"""

            logreg2 = LogisticRegression(C = 20.0,
                                         penalty = "l1")
            logreg2.fit(self.x_train, self.y_train)
            y_pred = logreg2.predict(self.x_test)

            acc_logreg2 = accuracy_score(y_pred, self.y_test)
            print("Logistic Regression Accuracy Score with Grid Search CV is : ", acc_logreg2)

            #Confusion Matrix
            conf_mat = confusion_matrix(self.y_test, y_pred)

            # ROC Curve
            predict_proba_logreg = logreg2.predict_proba(self.x_test)[::, 1]
            fpr, tpr, _ = metrics.roc_curve(self.y_test, predict_proba_logreg)
            auc_logreg = metrics.roc_auc_score(self.y_test, predict_proba_logreg)

            #Assign all score values to dict
            model_score_dict["accuracy"] = acc_logreg2
            model_score_dict["conf_mat"] = conf_mat.tolist()
            model_score_dict["fpr"] = fpr.tolist()
            model_score_dict["tpr"] = tpr.tolist()
            model_score_dict["auc"] = auc_logreg

            md = ModelDetail(**{'AlgorithmName': 'Logistic Regression', 'ModelScoreDict': str(model_score_dict)})
            md.save()

            # Export model
            with open('./HRAnalysis/analysemodels/models/LogReg.pkl', 'wb') as model_file:
                pickle.dump(logreg2, model_file)
        except Exception as e:
            raise e
