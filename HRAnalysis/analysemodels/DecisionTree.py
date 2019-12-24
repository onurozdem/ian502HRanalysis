import pickle
import datetime

from sklearn import metrics
import matplotlib.pyplot as plt
from HRAnalysis.models import ModelDetail
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


class DecisionTree:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        try:
            model_score_dict = dict()
            model_start_time = datetime.datetime.now()

            """dt = DecisionTreeClassifier()
            parameters = {"criterion": ["gini", "entropy"],
                          "max_depth": np.arange(1, 26, 1)}

            gridcv_dt = GridSearchCV(estimator=dt,
                                     param_grid=parameters,
                                     scoring="accuracy",
                                     cv=10)

            print("Grid Search started for Decision Tree: ", datetime.datetime.now())
            gridcv_dt.fit(self.x_train, self.y_train)
            print("Grid Search finished for Decision Tree: ", datetime.datetime.now())

            print("Best Parameters for Decision Tree are :", gridcv_dt.best_params_)
            print("accuracy :", gridcv_dt.best_score_)"""

            dt2 = DecisionTreeClassifier(criterion="entropy", max_depth=4)
            dt2.fit(self.x_train, self.y_train)
            y_pred = dt2.predict(self.x_test)
            acc_dt2 = accuracy_score(y_pred, self.y_test)
            print("Decision Tree Accuracy Score with Grid Search CV is : ", acc_dt2)

            model_end_time = datetime.datetime.now()
            model_running_performance = model_end_time - model_start_time

            #Confusion Matrix
            conf_mat = confusion_matrix(self.y_test, y_pred)

            # ROC Curve
            pred_proba_dt = dt2.predict_proba(self.x_test)[::, 1]
            fpr, tpr, _ = metrics.roc_curve(self.y_test, pred_proba_dt)
            auc_dt = metrics.roc_auc_score(self.y_test, pred_proba_dt)

            plt.figure()
            lw = 3
            plt.plot(fpr, tpr, label="Decision Tree, auc_dt = " + str(auc_dt))
            plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='dashed')
            plt.legend(loc=4)
            plt.savefig('./static/images/roc_dt.png')

            #Assign all score values to dict
            model_score_dict["model_running_performance"] = (model_running_performance.seconds/60)
            model_score_dict["accuracy"] = acc_dt2
            model_score_dict["conf_mat"] = conf_mat.tolist()
            model_score_dict["fpr"] = fpr.tolist()
            model_score_dict["tpr"] = tpr.tolist()
            model_score_dict["auc"] = auc_dt

            md = ModelDetail(**{'AlgorithmName': 'Decision Tree', 'ModelScoreDict': str(model_score_dict)})
            md.save()

            # Export model
            with open('./HRAnalysis/analysemodels/models/DecisionTree.pkl', 'wb') as model_file:
                pickle.dump(dt2, model_file)
        except Exception as e:
            raise e
