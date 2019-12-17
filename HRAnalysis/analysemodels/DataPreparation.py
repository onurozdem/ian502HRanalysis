import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np


class DataPreparation:
    def __init__(self, data):
        self.data = pd.DataFrame(data)

    def data_preparation(self):
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