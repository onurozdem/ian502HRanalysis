import pickle
import datetime

from sklearn import metrics
import matplotlib.pyplot as plt
from HRAnalysis.models import ModelDetail
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV


class Adaboost:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        try:
            model_score_dict = dict()
            model_start_time = datetime.datetime.now()

            """ada = AdaBoostClassifier()
            parameters = {
                "learning_rate": [0.01, 0.05, 0.1, 0.3, 1, 2],
                "n_estimators": [50, 100, 1000],
                "algorithm": ["SAMME", "SAMME.R"]
            }
            gridcv_ada = GridSearchCV(estimator=ada,
                                      param_grid=parameters,
                                      scoring='accuracy',
                                      cv=10)

            print("Grid Search started for Adaboost: ", datetime.datetime.now())
            gridcv_ada.fit(self.x_train, self.y_train)
            print("Grid Search finished for Adaboost: ", datetime.datetime.now())

            print("Best Parameters for Adaboost are :", gridcv_ada.best_params_)
            print("accuracy :", gridcv_ada.best_score_)"""

            ada2 = AdaBoostClassifier(algorithm="SAMME",
                                      learning_rate=0.3,
                                      n_estimators=1000)
            ada2.fit(self.x_train, self.y_train)
            y_pred = ada2.predict(self.x_test)
            acc_ada2 = accuracy_score(y_pred, self.y_test)
            print("Adaboost Accuracy Score with Grid Search CV is : ", acc_ada2)

            model_end_time = datetime.datetime.now()
            model_running_performance = model_end_time - model_start_time

            #Confusion Matrix
            conf_mat = confusion_matrix(self.y_test, y_pred)

            # ROC Curve
            pred_proba_ada = ada2.predict_proba(self.x_test)[::, 1]
            fpr, tpr, _ = metrics.roc_curve(self.y_test, pred_proba_ada)
            auc_ada = metrics.roc_auc_score(self.y_test, pred_proba_ada)

            plt.figure()
            lw = 3
            plt.plot(fpr, tpr, label="Adaboost, auc_ada = " + str(auc_ada))
            plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='dashed')
            plt.legend(loc=4)
            plt.savefig('./static/images/roc_ada.png')

            #Assign all score values to dict
            model_score_dict["model_running_performance"] = (model_running_performance.seconds/60)
            model_score_dict["accuracy"] = acc_ada2
            model_score_dict["conf_mat"] = conf_mat.tolist()
            model_score_dict["fpr"] = fpr.tolist()
            model_score_dict["tpr"] = tpr.tolist()
            model_score_dict["auc"] = auc_ada

            md = ModelDetail(**{'AlgorithmName': 'Adaboost', 'ModelScoreDict': str(model_score_dict)})
            md.save()

            # Export model
            with open('./HRAnalysis/analysemodels/models/Adaboost.pkl', 'wb') as model_file:
                pickle.dump(ada2, model_file)
        except Exception as e:
            raise e
