import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from HRAnalysis.analysemodels import Adaboost, DecisionTree, KNN, LinearDiscriminantAnalysis, \
                                     LogisticRegression, NaiveBayes, RandomForest, SVM, NeuralNetwork


class TrainModels:
    def __init__(self, data):
        if isinstance(data, list):
            self.data = pd.DataFrame(data)
        else:
            raise Exception("Train Models running unsuccessful. Data must be list type.")


        try:
            x_train, x_test, y_train, y_test = self.all_data_preparation()

            Adaboost.Adaboost(x_train, x_test, y_train, y_test).train()
            DecisionTree.DecisionTree(x_train, x_test, y_train, y_test).train()
            KNN.KNN(x_train, x_test, y_train, y_test).train()
            LinearDiscriminantAnalysis.LDA(x_train, x_test, y_train, y_test).train()
            LogisticRegression.LR(x_train, x_test, y_train, y_test).train()
            NaiveBayes.NaiveBayes(x_train, x_test, y_train, y_test).train()
            RandomForest.RandomForest(x_train, x_test, y_train, y_test).train()
            SVM.SVM(x_train, x_test, y_train, y_test).train()
            NeuralNetwork.NeuralNetwork(x_train, x_test, y_train, y_test).train()
        except Exception as e:
            raise e

    def all_data_preparation(self):
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
        data2 = self.data.apply(le.fit_transform)

        corr = pd.DataFrame(data2.corr())
        corr.fillna(0, inplace=True)

        rowLabel = list(corr)

        heatmap_data = []
        heatmap_data.append(rowLabel)
        heatmap_data = corr.__array__().tolist()

        with open('./static/data/heatmap.tsv', 'wt',  newline='') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(["row_idx", "col_idx", "log2ratio"])
            for column in range(len(heatmap_data)):
                for row in range(len(heatmap_data)):
                    tmp_list = []
                    tmp_list.append(row + 1)
                    tmp_list.append(column + 1)
                    tmp_list.append(heatmap_data[row][column])
                    tsv_writer.writerow(tmp_list)

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(left=0.15, top=0.94, bottom=0.1)
        sns.boxplot('Gender', 'Age', data=self.data[['Gender', 'Age']], ax=ax)
        plt.savefig('./static/images/box_plot_age_gender.png')

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(left=0.15, top=0.94, bottom=0.1)
        sns.boxplot('Gender', 'MonthlyIncome', data=self.data[['Gender', 'MonthlyIncome']], ax=ax)
        plt.savefig('./static/images/box_plot_income_gender.png')

        fig, ax = plt.subplots(figsize=(21, 6))
        fig.subplots_adjust(left=0.05, top=0.94, bottom=0.1)
        sns.boxplot('JobRole', 'MonthlyIncome', data=self.data[['JobRole', 'MonthlyIncome']], ax=ax)
        plt.savefig('./static/images/box_plot_income_role.png')

        fig, ax = plt.subplots(figsize=(16, 6))
        fig.subplots_adjust(left=0.05, top=0.94, bottom=0.1)
        sns.boxplot('DistanceFromHome', 'MonthlyIncome', data=self.data[['DistanceFromHome', 'MonthlyIncome']], ax=ax)
        plt.savefig('./static/images/box_plot_distance_income.png')

        # Spliting target column

        target_attrition = data2['Attrition']
        data3 = data2.drop(['Attrition'], axis=1)

        # Drop unnecessary columns
        # These columns have same value on rows. Therefore they are removed.
        data3 = data3.drop(['EmployeeCount','EmployeeNumber','Over18'], axis = 1)

        # Feature Selection
        rf = RandomForestClassifier()
        rf.fit(data3, target_attrition)

        feat_imp = pd.DataFrame()
        feat_imp["FeatName"] = data3.columns
        feat_imp["FeatImportance"] = pd.DataFrame(rf.feature_importances_, columns=["FeatImportance"])

        importance_thres = 0.02
        data4 = pd.DataFrame()

        for i in range(0, len(feat_imp)):
            if feat_imp["FeatImportance"][i] > importance_thres:
                column = feat_imp["FeatName"][i]
                data4[column] = data3[column]

        # Standardize Numeric Columns
        # Numeric Columns
        NumericCols = ['DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate',
                       'TotalWorkingYears',
                       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

        # 'DailyRate', 'DistanceFromHome' 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'TotalWorkingYears',
        # 'YearsAtCompany' , 'YearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager'

        StSc = StandardScaler()
        for i in range(0, len(data4.columns)):
            if data4.columns[i] in NumericCols:
                print(data4.columns[i])
                data4[data4.columns[i]] = pd.DataFrame(StSc.fit_transform(data4.iloc[:, i:(i + 1)]))

        # Data splitting
        x_train, x_test, y_train, y_test = train_test_split(data4,
                                                            target_attrition,
                                                            test_size = 0.2,
                                                            random_state = 0)
        return x_train, x_test, y_train, y_test
