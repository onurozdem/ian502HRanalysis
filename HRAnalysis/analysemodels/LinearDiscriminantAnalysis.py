import pickle
import datetime

from sklearn import metrics
import matplotlib.pyplot as plt
from HRAnalysis.models import ModelDetail
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        try:
            model_score_dict = dict()
            model_start_time = datetime.datetime.now()

            lda = LinearDiscriminantAnalysis(shrinkage="auto",
                                             solver="lsqr",  # eigen, svd(default)
                                             )
            lda.fit(self.x_train, self.y_train)
            y_pred = lda.predict(self.x_test)
            acc_lda = accuracy_score(y_pred, self.y_test)
            print("Linear Discriminant Analysis Accuracy Score is : ", acc_lda)

            model_end_time = datetime.datetime.now()
            model_running_performance = model_end_time - model_start_time

            #Confusion Matrix
            conf_mat = confusion_matrix(self.y_test, y_pred)

            # ROC Curve
            pred_proba_lda = lda.predict_proba(self.x_test)[::, 1]
            fpr, tpr, _ = metrics.roc_curve(self.y_test, pred_proba_lda)
            auc_lda = metrics.roc_auc_score(self.y_test, pred_proba_lda)

            plt.figure()
            lw = 3
            plt.plot(fpr, tpr, label="Linear Discriminant Analysis, auc_lda = " + str(auc_lda))
            plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='dashed')
            plt.legend(loc=4)
            plt.savefig('./static/images/roc_lda.png')

            #Assign all score values to dict
            model_score_dict["model_running_performance"] = (model_running_performance.seconds/60)
            model_score_dict["accuracy"] = acc_lda
            model_score_dict["conf_mat"] = conf_mat.tolist()
            model_score_dict["fpr"] = fpr.tolist()
            model_score_dict["tpr"] = tpr.tolist()
            model_score_dict["auc"] = auc_lda

            md = ModelDetail(**{'AlgorithmName': 'Linear Discriminant Analysis', 'ModelScoreDict': str(model_score_dict)})
            md.save()

            # Export model
            with open('./HRAnalysis/analysemodels/models/LDA.pkl', 'wb') as model_file:
                pickle.dump(lda, model_file)
        except Exception as e:
            raise e
