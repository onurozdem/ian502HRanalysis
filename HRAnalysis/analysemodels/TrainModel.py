import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from HRAnalysis.analysemodels import Adaboost, DecisionTree, KNN, LinearDiscriminantAnalysis, LogisticRegression, NaiveBayes, RandomForest, SVM


class TrainModels:
    def __init__(self, data):
        if isinstance(data, list):
            self.data = pd.DataFrame(data)
        else:
            raise Exception("Train Models running unsuccessful. Data must be list type.")

        x_train, x_test, y_train, y_test = self.all_data_preparation()
        try:
            Adaboost.Adaboost(x_train, x_test, y_train, y_test).train()
            DecisionTree.DecisionTree(x_train, x_test, y_train, y_test).train()
            KNN.KNN(x_train, x_test, y_train, y_test).train()
            LinearDiscriminantAnalysis.LDA(x_train, x_test, y_train, y_test).train()
            LogisticRegression.LR(x_train, x_test, y_train, y_test).train()
            NaiveBayes.NaiveBayes(x_train, x_test, y_train, y_test).train()
            RandomForest.RandomForest(x_train, x_test, y_train, y_test).train()
            SVM.SVM(x_train, x_test, y_train, y_test).train()
        except Exception as e:
            raise e

    def all_data_preparation(self):
        # These columns have same value on rows. Therefore they are removed.
        data = self.data.drop(['EmployeeCount','EmployeeNumber','Over18'], axis = 1)

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


        # Spliting target column
        target_attrition = data2['Attrition']
        data3 = data2.drop(['Attrition'], axis = 1)

        # Modeling
        x_train, x_test, y_train, y_test = train_test_split(data3,
                                                            target_attrition,
                                                            test_size = 0.2,
                                                            random_state = 0)
        return x_train, x_test, y_train, y_test