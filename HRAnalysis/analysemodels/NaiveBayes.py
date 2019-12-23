import pickle
import datetime

from sklearn import metrics
from HRAnalysis.models import ModelDetail
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


class NaiveBayes:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        try:
            model_score_dict = dict()
            model_start_time = datetime.datetime.now()

            nb = GaussianNB()
            nb.fit(self.x_train, self.y_train)
            y_pred = nb.predict(self.x_test)
            acc_nb = accuracy_score(y_pred, self.y_test)
            print("Naive Bayes Accuracy Score is : ", acc_nb)

            model_end_time = datetime.datetime.now()
            model_running_performance = model_end_time - model_start_time

            #Confusion Matrix
            conf_mat = confusion_matrix(self.y_test, y_pred)

            # ROC Curve
            pred_proba_nb = nb.predict_proba(self.x_test)[::, 1]
            fpr, tpr, _ = metrics.roc_curve(self.y_test, pred_proba_nb)
            auc_nb = metrics.roc_auc_score(self.y_test, pred_proba_nb)

            #Assign all score values to dict
            model_score_dict["model_running_performance"] = (model_running_performance.seconds/60)
            model_score_dict["accuracy"] = acc_nb
            model_score_dict["conf_mat"] = conf_mat.tolist()
            model_score_dict["fpr"] = fpr.tolist()
            model_score_dict["tpr"] = tpr.tolist()
            model_score_dict["auc"] = auc_nb

            md = ModelDetail(**{'AlgorithmName': 'Naive Bayes', 'ModelScoreDict': str(model_score_dict)})
            md.save()

            # Export model
            with open('./HRAnalysis/analysemodels/models/NB.pkl', 'wb') as model_file:
                pickle.dump(nb, model_file)
        except Exception as e:
            raise e
