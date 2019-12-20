import pickle
import datetime

from sklearn import metrics
from HRAnalysis.models import ModelDetail
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        try:
            model_score_dict = dict()

            """rf = RandomForestClassifier()
            
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
            print("accuracy :",gridcv_rf.best_score_)"""

            rf2 = RandomForestClassifier(criterion = "entropy",
                                         max_depth = 8,
                                         max_features = "auto",
                                         n_estimators = 100)
            rf2.fit(self.x_train, self.y_train)
            y_pred = rf2.predict(self.x_test)
            acc_rf2 = accuracy_score(y_pred, self.y_test)
            print("Random Forest Accuracy Score with Grid Search CV is : ", acc_rf2)

            #Confusion Matrix
            conf_mat = confusion_matrix(self.y_test, y_pred)

            # ROC Curve
            pred_proba_rf = rf2.predict_proba(self.x_test)[::, 1]
            fpr, tpr, _ = metrics.roc_curve(self.y_test, pred_proba_rf)
            auc_rf = metrics.roc_auc_score(self.y_test, pred_proba_rf)

            #Assign all score values to dict
            model_score_dict["accuracy"] = acc_rf2
            model_score_dict["conf_mat"] = conf_mat.tolist()
            model_score_dict["fpr"] = fpr.tolist()
            model_score_dict["tpr"] = tpr.tolist()
            model_score_dict["auc"] = auc_rf

            md = ModelDetail(**{'AlgorithmName': 'Random Forest', 'ModelScoreDict': str(model_score_dict)})
            md.save()

            # Export model
            with open('./HRAnalysis/analysemodels/models/RF.pkl', 'wb') as model_file:
                pickle.dump(rf2, model_file)
        except Exception as e:
            raise e
