import h5py
import json
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from HRAnalysis.models import *
from sklearn.preprocessing import LabelEncoder
from HRAnalysis.analysemodels import TrainModel
from django.shortcuts import render, HttpResponse


def index(request):
    data = None
    if request.method == 'GET':
        row_number = UnprocessedData.objects.count()
        logistic_regr = ModelDetail.objects.filter(AlgorithmName='Logistic Regression').order_by('-Date').first()
        dt = ModelDetail.objects.filter(AlgorithmName='Decision Tree').order_by('-Date').first()
        rf = ModelDetail.objects.filter(AlgorithmName='Random Forest').order_by('-Date').first()
        ada = ModelDetail.objects.filter(AlgorithmName='Adaboost').order_by('-Date').first()
        knn = ModelDetail.objects.filter(AlgorithmName='KNN').order_by('-Date').first()
        lda = ModelDetail.objects.filter(AlgorithmName='Linear Discriminant Analysis').order_by('-Date').first()
        nb = ModelDetail.objects.filter(AlgorithmName='Naive Bayes').order_by('-Date').first()
        ann = ModelDetail.objects.filter(AlgorithmName='ANN').order_by('-Date').first()
        svm = ModelDetail.objects.filter(AlgorithmName='SVM').order_by('-Date').first()
        data = {'row_number': row_number,
                'dt': str(json.loads(dt.ModelScoreDict.replace("'",'"'))["accuracy"]),
                'logistic_regression': str(json.loads(logistic_regr.ModelScoreDict.replace("'",'"'))["accuracy"]),
                'ada': str(json.loads(ada.ModelScoreDict.replace("'",'"'))["accuracy"]),
                'knn': str(json.loads(knn.ModelScoreDict.replace("'",'"'))["accuracy"]),
                'lda': str(json.loads(lda.ModelScoreDict.replace("'",'"'))["accuracy"]),
                'nb': str(json.loads(nb.ModelScoreDict.replace("'",'"'))["accuracy"]),
                'ann': str(json.loads(ann.ModelScoreDict.replace("'",'"'))["accuracy"]),
                'svm': str(json.loads(svm.ModelScoreDict.replace("'",'"'))["accuracy"]),
                'rf': str(json.loads(rf.ModelScoreDict.replace("'",'"'))["accuracy"])}
    return render(request, 'index.html', {'data': data})


def about(request):
    return render(request, 'about.html', {})


def project_members(request):
    return render(request, 'project_members.html', {})


def model_compare(request):
    check_require_train()

    data1 = {}
    data2 = {}
    if request.method == 'POST':
        model_compare_form = ModelCompareFormData(request.POST)

        if "algorithm1" in model_compare_form.pk.keys() and "algorithm2" in model_compare_form.pk.keys():
            long_algorithm_name1 = convert_algorithm_name(model_compare_form.pk['algorithm1'])
            long_algorithm_name2 = convert_algorithm_name(model_compare_form.pk['algorithm2'])

            model_detail_object1 = ModelDetail.objects.filter(AlgorithmName=long_algorithm_name1)
            model_detail_object2 = ModelDetail.objects.filter(AlgorithmName=long_algorithm_name2)

            model_detail_row1 = model_detail_object1.order_by('-Date').first()
            model_detail_row2 = model_detail_object2.order_by('-Date').first()

            data1['conf_mat'] = json.loads(model_detail_row1.ModelScoreDict.replace("'", '"'))["conf_mat"]
            data2['conf_mat'] = json.loads(model_detail_row2.ModelScoreDict.replace("'", '"'))["conf_mat"]
            data1['algorithm'] = long_algorithm_name1
            data2['algorithm'] = long_algorithm_name2
            data1['short_algorithm'] = model_compare_form.pk['algorithm1']
            data2['short_algorithm'] = model_compare_form.pk['algorithm2']
            data1['roc_file'] = get_roc_file_name(model_compare_form.pk['algorithm1'])
            data2['roc_file'] = get_roc_file_name(model_compare_form.pk['algorithm2'])

            model_detail1 = model_detail_object1.order_by('Date').all()
            model_detail2 = model_detail_object2.order_by('Date').all()
            accuracy_list1 = []
            accuracy_list2 = []
            performance_list1 = []
            performance_list2 = []
            date_list1 = []
            date_list2 = []

            for i in model_detail1:
                accuracy_list1.append(json.loads(i.ModelScoreDict.replace("'", '"'))["accuracy"])
                performance_list1.append(json.loads(i.ModelScoreDict.replace("'", '"'))["model_running_performance"])
                date_list1.append(i.Date.strftime("%d/%m/%y %H:%M"))

            for i in model_detail2:
                accuracy_list2.append(json.loads(i.ModelScoreDict.replace("'", '"'))["accuracy"])
                performance_list2.append(json.loads(i.ModelScoreDict.replace("'", '"'))["model_running_performance"])
                date_list2.append(i.Date.strftime("%d/%m/%y %H:%M"))

            plt_df1 = pd.DataFrame({'xvalues': date_list1, 'yvalues': accuracy_list1})
            # plot
            plt.figure()
            plt.plot('xvalues', 'yvalues', data=plt_df1)
            plt.title('Model Accuracy History')
            plt.ylabel('Accuracy')
            plt.xlabel('Date Time')
            plt.savefig('./static/images/line_{}.png'.format(data1['short_algorithm']))

            plt_df2 = pd.DataFrame({'xvalues': date_list2, 'yvalues': accuracy_list2})
            # plot
            plt.figure()
            plt.plot('xvalues', 'yvalues', data=plt_df2)
            plt.title('Model Accuracy History')
            plt.ylabel('Accuracy')
            plt.xlabel('Date Time')
            plt.savefig('./static/images/line_{}.png'.format(data2['short_algorithm']))

            plt_df1 = pd.DataFrame({'xvalues': date_list1, 'yvalues': performance_list1})
            # plot
            plt.figure()
            plt.plot('xvalues', 'yvalues', data=plt_df1)
            plt.title('Model Running Performance History')
            plt.ylabel('Time(minute)')
            plt.xlabel('Date Time')
            plt.savefig('./static/images/line_perf_{}.png'.format(data1['short_algorithm']))

            plt_df2 = pd.DataFrame({'xvalues': date_list2, 'yvalues': performance_list2})
            # plot
            plt.figure()
            plt.plot('xvalues', 'yvalues', data=plt_df2)
            plt.title('Model Running Performance History')
            plt.ylabel('Time(minute)')
            plt.xlabel('Date Time')
            plt.savefig('./static/images/line_perf_{}.png'.format(data2['short_algorithm']))

            data1['line_file'] = 'line_{}.png'.format(data1['short_algorithm'])
            data2['line_file'] = 'line_{}.png'.format(data2['short_algorithm'])
            data1['line_perf_file'] = 'line_perf_{}.png'.format(data1['short_algorithm'])
            data2['line_perf_file'] = 'line_perf_{}.png'.format(data2['short_algorithm'])
            data1['is_analysed'] = True
            data2['is_analysed'] = True

    else:
        model_detail_form = ModelDetailFormData()
        data1['short_algorithm'] = "adaboost"
        data2['short_algorithm'] = "adaboost"

    return render(request, 'model_compare.html', {'data1': data1, 'data2': data2})


def model_detail(request):
    check_require_train()

    data = {}
    if request.method == 'POST':
        model_detail_form = ModelDetailFormData(request.POST)

        if "algorithm" in model_detail_form.pk.keys():
            long_algorithm_name = convert_algorithm_name(model_detail_form.pk['algorithm'])
            model_detail_object = ModelDetail.objects.filter(AlgorithmName=long_algorithm_name)

            model_detail_row = model_detail_object.order_by('-Date').first()

            data['conf_mat'] = json.loads(model_detail_row.ModelScoreDict.replace("'", '"'))["conf_mat"]
            data['algorithm'] = long_algorithm_name
            data['short_algorithm'] = model_detail_form.pk['algorithm']
            data['roc_file'] = get_roc_file_name(model_detail_form.pk['algorithm'])

            model_detail = model_detail_object.order_by('Date').all()
            accuracy_list = []
            performance_list = []
            date_list = []
            for i in model_detail:
                accuracy_list.append(json.loads(i.ModelScoreDict.replace("'", '"'))["accuracy"])
                performance_list.append(json.loads(i.ModelScoreDict.replace("'", '"'))["model_running_performance"])
                date_list.append(i.Date.strftime("%d/%m/%y %H:%M"))

            plt_df = pd.DataFrame({'xvalues': date_list, 'yvalues': accuracy_list})
            # plot
            plt.figure()
            plt.plot('xvalues', 'yvalues', data=plt_df)
            plt.title('Model Accuracy History')
            plt.ylabel('Accuracy')
            plt.xlabel('Date Time')
            plt.savefig('./static/images/line_{}.png'.format(data['short_algorithm']))

            plt_df = pd.DataFrame({'xvalues': date_list, 'yvalues': performance_list})
            # plot
            plt.figure()
            plt.plot('xvalues', 'yvalues', data=plt_df)
            plt.title('Model Running Performance History')
            plt.ylabel('Time(minute)')
            plt.xlabel('Date Time')
            plt.savefig('./static/images/line_perf_{}.png'.format(data['short_algorithm']))

            data['line_file'] = 'line_{}.png'.format(data['short_algorithm'])
            data['line_perf_file'] = 'line_perf_{}.png'.format(data['short_algorithm'])
            data['is_analysed'] = True
    else:
        model_detail_form = ModelDetailFormData()
        data['short_algorithm'] = "adaboost"

    return render(request, 'model_detail.html', {'data': data})


def data_detail(request):
    check_require_train()
    table_data = []
    columns = ["Age", "Attrition", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome", "Education",
               "EducationField", "EmployeeCount", "EmployeeNumber", "EnvironmentSatisfaction", "Gender", "HourlyRate",
               "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome",
               "MonthlyRate",
               "NumCompaniesWorked", "Over18", "OverTime", "PercentSalaryHike", "PerformanceRating",
               "RelationshipSatisfaction", "StandardHours", "StockOptionLevel", "TotalWorkingYears",
               "TrainingTimesLastYear",
               "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
               "YearsWithCurrManager"]

    for row in UnprocessedData.objects.values():
        tmp_list = []
        for column in columns:
            if row[column] is not None:
                tmp_list.append(row[column])
            else:
                tmp_list.append('null')

        table_data.append(tmp_list)

    data = {"table_data": table_data}
    return render(request, 'data_detail.html', data)


def predict_data(request):
    check_require_train()
    response_data = {}
    if request.method == 'POST':
        predict_form = PredictFormData(request.POST)
        if "algorithm" in predict_form.pk.keys():
            algorithm = predict_form.pk['algorithm']

            data = {}
            data['Age'] = predict_form.pk['Age']
            data['DailyRate'] = predict_form.pk['DailyRate']
            data['DistanceFromHome'] = predict_form.pk['DistanceFromHome']
            data['EducationField'] = predict_form.pk['EducationField']
            data['EnvironmentSatisfaction'] = predict_form.pk['EnvironmentSatisfaction']
            data['HourlyRate'] = predict_form.pk['HourlyRate']
            data['JobInvolvement'] = predict_form.pk['JobInvolvement']
            data['JobLevel'] = predict_form.pk['JobLevel']
            data['JobRole'] = predict_form.pk['JobRole']
            data['JobSatisfaction'] = predict_form.pk['JobSatisfaction']
            data['MonthlyIncome'] = predict_form.pk['MonthlyIncome']
            data['MonthlyRate'] = predict_form.pk['MonthlyRate']
            data['NumCompaniesWorked'] = predict_form.pk['NumCompaniesWorked']
            data['OverTime'] = predict_form.pk['OverTime']
            data['PercentSalaryHike'] = predict_form.pk['PercentSalaryHike']
            data['RelationshipSatisfaction'] = predict_form.pk['RelationshipSatisfaction']
            data['StockOptionLevel'] = predict_form.pk['StockOptionLevel']
            data['TotalWorkingYears'] = predict_form.pk['TotalWorkingYears']
            data['TrainingTimesLastYear'] = predict_form.pk['TrainingTimesLastYear']
            data['WorkLifeBalance'] = predict_form.pk['WorkLifeBalance']
            data['YearsAtCompany'] = predict_form.pk['YearsAtCompany']
            data['YearsInCurrentRole'] = predict_form.pk['YearsInCurrentRole']
            data['YearsSinceLastPromotion'] = predict_form.pk['YearsSinceLastPromotion']
            data['YearsWithCurrManager'] = predict_form.pk['YearsWithCurrManager']
            data['BusinessTravel'] = predict_form.pk['BusinessTravel']
            data['Department'] = predict_form.pk['Department']
            data['Education'] = predict_form.pk['Education']
            data['Gender'] = predict_form.pk['Gender']
            data['MaritalStatus'] = predict_form.pk['MaritalStatus']
            data['PerformanceRating'] = predict_form.pk['PerformanceRating']
            data['StandardHours'] = predict_form.pk['StandardHours']

            model_file_name = get_model_file_name(algorithm)

            if algorithm == "ann":
                import tensorflow.keras as kr

                model_file_data = {}
                model_file_data["model"] = kr.models.load_model('./HRAnalysis/analysemodels/models/{}'.format(model_file_name))

                with open('./HRAnalysis/analysemodels/models/ann.txt', 'r') as f:
                    model_file_data["columns"] = json.load(f)["columns"]
            else:
                with open('./HRAnalysis/analysemodels/models/{}'.format(model_file_name), 'rb') as f:
                    model_file_data = pickle.load(f)

            predict_data_dict = {}
            for i in model_file_data["columns"]:
                predict_data_dict[i] = data[i]
                response_data[i] = data[i]

            predict_data = pd.DataFrame(predict_data_dict, index=[0])

            le = LabelEncoder()
            data2 = predict_data.apply(le.fit_transform)

            y_pred = model_file_data["model"].predict(data2)

            data['Attrition'] = y_pred
            udm = UnprocessedData(**data)
            udm.save()

            if int(y_pred[0]) == 1:
                response_data["is_attrition"] = "Employee will be attrition."
            else:
                response_data["is_attrition"] = "Employee won't be attrition."

            response_data['is_analysed'] = True
            response_data['short_algorithm'] = algorithm
    else:
        predict_form = PredictFormData()
        response_data['short_algorithm'] = "adaboost"
        response_data["EnvironmentSatisfaction"] = "1"
        response_data["JobInvolvement"] = "1"
        response_data["Education"] = "1"
        response_data["PerformanceRating"] = "1"
        response_data["JobSatisfaction"] = "1"
        response_data["RelationshipSatisfaction"] = "1"
        response_data["WorkLifeBalance"] = "1"
        response_data["JobRole"] = "Healthcare Representative"
        response_data["MaritalStatus"] = "Single"
        response_data["Gender"] = "Male"
        response_data["EducationField"] = "Human Resources"
        response_data["Department"] = "Sales"
        response_data["BusinessTravel"] = "Travel_Rarely"

    return render(request, 'predict_data.html', {'data':response_data})


def contact(request):
    return render(request, 'contact.html', {})


def check_require_train():
    if ModelDetail.objects.count() > 0:
        last_train_date = ModelDetail.objects.order_by('Date').first()
        last_train_date_check = (datetime.datetime.now() - last_train_date.Date.replace(tzinfo=None)).days
    else:
        last_train_date_check = 7

    if last_train_date_check >= 7:
        analyse_data = []
        for i in UnprocessedData.objects.values():
            analyse_data.append(i)

        try:
            TrainModel.TrainModels(analyse_data)
        except:
            raise
    else:
        print("No need train model!")


def convert_algorithm_name(name):
    long_name = None

    if name == "adaboost":
        long_name = "Adaboost"
    elif name == "dt":
        long_name = "Decision Tree"
    elif name == "knn":
        long_name = "KNN"
    elif name == "lda":
        long_name = "Linear Discriminant Analysis"
    elif name == "lgr":
        long_name = "Logistic Regression"
    elif name == "nb":
        long_name = "Naive Bayes"
    elif name == "ann":
        long_name = "ANN"
    elif name == "rf":
        long_name = "Random Forest"
    elif name == "svm":
        long_name = "SVM"

    return long_name


def get_roc_file_name(name):
    file_name = None

    if name == "adaboost":
        file_name = "roc_ada.png"
    elif name == "dt":
        file_name = "roc_dt.png"
    elif name == "knn":
        file_name = "roc_knn.png"
    elif name == "lda":
        file_name = "roc_lda.png"
    elif name == "lgr":
        file_name = "roc_logr.png"
    elif name == "nb":
        file_name = "roc_nb.png"
    elif name == "ann":
        file_name = "roc_ann.png"
    elif name == "rf":
        file_name = "roc_rf.png"
    elif name == "svm":
        file_name = "roc_svm.png"

    return file_name


def get_model_file_name(name):
    file_name = None

    if name == "adaboost":
        file_name = "Adaboost.pkl"
    elif name == "dt":
        file_name = "DecisionTree.pkl"
    elif name == "knn":
        file_name = "KNN.pkl"
    elif name == "lda":
        file_name = "LDA.pkl"
    elif name == "lgr":
        file_name = "LogReg.pkl"
    elif name == "nb":
        file_name = "NB.pkl"
    elif name == "ann":
        file_name = "ANN.h5"
    elif name == "rf":
        file_name = "RF.pkl"
    elif name == "svm":
        file_name = "SVM.pkl"

    return file_name
