import datetime
import csv
import pandas as pd
import json

from HRAnalysis.models import *
from HRAnalysis.forms import PredictForm
from django.forms.models import model_to_dict
from sklearn.preprocessing import LabelEncoder
from HRAnalysis.analysemodels import TrainModel
from django.shortcuts import render, reverse, HttpResponseRedirect


def index(request):
    data = None
    if request.method == 'GET':
        row_number = UnprocessedData.objects.count()
        linear_regr = ModelDetail.objects.filter(AlgorithmName='Logistic Regression').order_by('Date').first()
        logistic_regr = ModelDetail.objects.filter(AlgorithmName='Decision Tree').order_by('Date').first()
        knn = ModelDetail.objects.filter(AlgorithmName='Random Forest').order_by('Date').first()
        data = {'row_number': row_number,
                'linear_regression': str(json.loads(linear_regr.ModelScoreDict.replace("'",'"'))["accuracy"]),
                'logistic_regression': str(json.loads(logistic_regr.ModelScoreDict.replace("'",'"'))["accuracy"]),
                'knn': str(json.loads(knn.ModelScoreDict.replace("'",'"'))["accuracy"])}
    return render(request, 'index.html', {'data': data})


def about(request):
    return render(request, 'about.html', {})


def project_members(request):
    return render(request, 'project_members.html', {})


def model_compare(request):
    return render(request, 'model_compare.html', {})


def model_detail(request):
    check_require_train()

    data = {}
    """if request.GET:
        data = request.GET
        #data = PredictForm.objects.filter(query__icontains=search)

        #name = request.GET.get('algorithm')
        #query = PredictForm.object.create(query=search, user_id=name)
        #query.save()"""
    if request.method == 'POST':
        model_detail_form = ModelDetailFormData(request.POST)
        long_algorithm_name = convert_algorithm_name(model_detail_form.pk['algorithm'])
        if "algorithm" in model_detail_form.pk.keys():
            model_detail_row = ModelDetail.objects.filter(AlgorithmName=long_algorithm_name).order_by('Date').first()
            data['conf_mat'] = json.loads(model_detail_row.ModelScoreDict.replace("'", '"'))["conf_mat"]
            data['algorithm'] = long_algorithm_name
            data['is_analysed'] = True
    else:
        model_detail_form = ModelDetailFormData()

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
            tmp_list.append(row[column])

        table_data.append(tmp_list)

    data = {"table_data": table_data}
    return render(request, 'data_detail.html', data)


def predict_data(request):
    check_require_train()
    data = {}
    """if request.GET:
        data = request.GET
        #data = PredictForm.objects.filter(query__icontains=search)

        #name = request.GET.get('algorithm')
        #query = PredictForm.object.create(query=search, user_id=name)
        #query.save()"""
    if request.method == 'POST':
        predict_form = PredictForm(request.POST)
        if "algorithm" in predict_form.data.keys():
            print(1)
            data['fname'] = predict_form.data['fname']
            #return HttpResponseRedirect(reverse('form-redirect'))
            data['is_analysed'] = True
    else:
        predict_form = PredictForm()

    return render(request, 'predict_data.html', {'data':data, 'predict_form': predict_form})


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
        long_name = "Linear Discriminant Anaylsis"
    elif name == "lgr":
        long_name = "Lojistic Regression"
    elif name == "nb":
        long_name = "Naive Bayes"
    elif name == "ann":
        long_name = "ANN"
    elif name == "rf":
        long_name = "Random Forest"
    elif name == "svm":
        long_name = "SVM"

    return long_name


