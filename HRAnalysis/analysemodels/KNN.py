import pickle
import datetime

from sklearn import metrics
import matplotlib.pyplot as plt
from HRAnalysis.models import ModelDetail
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        try:
            model_score_dict = dict()
            model_start_time = datetime.datetime.now()

            """knn = KNeighborsClassifier()

            parameters = {"leaf_size": np.arange(2, 20, 1)}
            gridcv_knn = GridSearchCV(estimator=knn,
                                      param_grid=parameters,
                                      scoring="accuracy",
                                      cv=10)

            print("Grid Search started for KNN : ", datetime.datetime.now())
            gridcv_knn.fit(self.x_train, self.y_train)
            print("Grid Search finished for KNN : ", datetime.datetime.now())

            print("Best Parameters for KNN are :", gridcv_knn.best_params_)
            print("accuracy :", gridcv_knn.best_score_)"""

            knn2 = KNeighborsClassifier(leaf_size=2)
            knn2.fit(self.x_train, self.y_train)
            y_pred = knn2.predict(self.x_test)
            acc_knn2 = accuracy_score(y_pred, self.y_test)
            print("KNN Accuracy Score is :", acc_knn2)

            model_end_time = datetime.datetime.now()
            model_running_performance = model_end_time - model_start_time

            #Confusion Matrix
            conf_mat = confusion_matrix(self.y_test, y_pred)

            # ROC Curve
            pred_proba_knn = knn2.predict_proba(self.x_test)[::, 1]
            fpr, tpr, _ = metrics.roc_curve(self.y_test, pred_proba_knn)
            auc_knn = metrics.roc_auc_score(self.y_test, pred_proba_knn)

            plt.figure()
            lw = 3
            plt.plot(fpr, tpr, label="KNN, auc_knn = " + str(auc_knn))
            plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='dashed')
            plt.title('KNN ROC')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc=4)
            plt.savefig('./static/images/roc_knn.png')

            #Assign all score values to dict
            model_score_dict["model_running_performance"] = (model_running_performance.seconds/60)
            model_score_dict["accuracy"] = acc_knn2
            model_score_dict["conf_mat"] = conf_mat.tolist()
            model_score_dict["fpr"] = fpr.tolist()
            model_score_dict["tpr"] = tpr.tolist()
            model_score_dict["auc"] = auc_knn

            md = ModelDetail(**{'AlgorithmName': 'KNN', 'ModelScoreDict': str(model_score_dict)})
            md.save()

            # Export model
            with open('./HRAnalysis/analysemodels/models/KNN.pkl', 'wb') as model_file:
                #pickle.dump(knn2, model_file)
                pickle.dump({"columns": self.x_test.columns.tolist(), "model": knn2}, model_file)
        except Exception as e:
            raise e
