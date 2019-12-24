import pickle
import datetime

from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
from HRAnalysis.models import ModelDetail
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


class SVM:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        try:
            model_score_dict = dict()
            model_start_time = datetime.datetime.now()

            svm = SVC()
            """
            parameters = {"C": np.arange(100, 1000, 100),
                          "gamma": [0.01, 0.001, 0.0001],
                          "kernel": ["rbf"]}

            gridcv_svm = GridSearchCV(estimator=svm,
                                      param_grid=parameters,
                                      scoring="accuracy",
                                      cv=10)

            print("Grid Search started for SVM: ", datetime.datetime.now())
            gridcv_svm.fit(x_train, y_train)
            print("Grid Search finished for SVM: ", datetime.datetime.now())

            print("Best Parameters for SVM are :", gridcv_svm.best_params_)
            print("accuracy :", gridcv_svm.best_score_)"""

            svm2 = SVC(C=100,
                       gamma=0.001,
                       kernel="rbf",
                       probability=True)
            svm2.fit(self.x_train, self.y_train)
            y_pred = svm2.predict(self.x_test)
            acc_svm2 = accuracy_score(y_pred, self.y_test)
            print("SVM Score with Grid Search CV is :", acc_svm2)

            model_end_time = datetime.datetime.now()
            model_running_performance = model_end_time - model_start_time

            #Confusion Matrix
            conf_mat = confusion_matrix(self.y_test, y_pred)

            # ROC Curve
            pred_proba_svm = svm2.predict_proba(self.x_test)[::, 1]
            fpr, tpr, _ = metrics.roc_curve(self.y_test, pred_proba_svm)
            auc_svm = metrics.roc_auc_score(self.y_test, pred_proba_svm)

            plt.figure()
            lw = 3
            plt.plot(fpr, tpr, label="Support Vector Machine, auc_svm = " + str(auc_svm))
            plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='dashed')
            plt.legend(loc=4)
            plt.savefig('./static/images/roc_svm.png')

            #Assign all score values to dict
            model_score_dict["model_running_performance"] = (model_running_performance.seconds/60)
            model_score_dict["accuracy"] = acc_svm2
            model_score_dict["conf_mat"] = conf_mat.tolist()
            model_score_dict["fpr"] = fpr.tolist()
            model_score_dict["tpr"] = tpr.tolist()
            model_score_dict["auc"] = auc_svm

            md = ModelDetail(**{'AlgorithmName': 'SVM', 'ModelScoreDict': str(model_score_dict)})
            md.save()

            # Export model
            with open('./HRAnalysis/analysemodels/models/SVM.pkl', 'wb') as model_file:
                pickle.dump(svm2, model_file)
        except Exception as e:
            raise e
