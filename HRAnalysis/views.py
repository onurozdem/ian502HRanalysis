from HRAnalysis.models import *
from django.shortcuts import render, reverse, HttpResponseRedirect
from HRAnalysis.forms import PredictForm
from HRAnalysis.analysemodels import DataPreparation
from django.forms.models import model_to_dict


def index(request):
    data = None
    if request.method == 'GET':
        row_number = UnprocessedData.objects.count()
        linear_regr = ModelDetail.objects.filter(AlgorithmName='LinearRegression').order_by('Date').first()
        logistic_regr = ModelDetail.objects.filter(AlgorithmName='LogisticRegression').order_by('Date').first()
        knn = ModelDetail.objects.filter(AlgorithmName='KNN').order_by('Date').first()
        data = {'row_number': row_number,
                'linear_regression': linear_regr,
                'logistic_regression': logistic_regr,
                'knn': knn}
    return render(request, 'index.html', {'data': data})


def about(request):
    return render(request, 'about.html', {})


def project_members(request):
    return render(request, 'project_members.html', {})


def model_compare(request):
    return render(request, 'model_compare.html', {})


def model_detail(request):
    analyse_data = []
    for i in UnprocessedData.objects.values():
        analyse_data.append(i)

    a,b,c,d = DataPreparation.DataPreparation(analyse_data).data_preparation()

    xdata = ["Apple", "Apricot", "Avocado", "Banana", "Boysenberries", "Blueberries", "Dates", "Grapefruit", "Kiwi",
             "Lemon"]
    ydata = [52, 48, 160, 94, 75, 71, 490, 82, 46, 17]
    chartdata = {'x': xdata, 'y': ydata}
    charttype = "pieChart"
    chartcontainer = 'piechart_container'
    data = {
        'charttype': charttype,
        'chartdata': chartdata,
        'chartcontainer': chartcontainer,
        'extra': {
            'x_is_date': False,
            'x_axis_format': '',
            'tag_script_js': True,
            'jquery_on_ready': False,
        },
        'rowLabel':["1","r","3","5"]
    }
    return render(request, 'model_detail.html', data)


def data_detail(request):
    xdata = ["Apple", "Apricot", "Avocado", "Banana", "Boysenberries", "Blueberries", "Dates", "Grapefruit", "Kiwi",
             "Lemon"]
    ydata = [52, 48, 160, 94, 75, 71, 490, 82, 46, 17]
    chartdata = {'x': xdata, 'y': ydata}
    charttype = "pieChart"
    chartcontainer = 'piechart_container'
    data = {
        'charttype': charttype,
        'chartdata': chartdata,
        'chartcontainer': chartcontainer,
        'extra': {
            'x_is_date': False,
            'x_axis_format': '',
            'tag_script_js': True,
            'jquery_on_ready': False,
        },
        'rowLabel':["1","r","3","5"]
    }
    return render(request, 'model_detail.html', data)


def predict_data(request):
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