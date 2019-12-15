from HRAnalysis.models import *
from django.shortcuts import render, reverse, HttpResponseRedirect
from HRAnalysis.forms import PredictForm


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
    return render(request, 'model_detail.html', {})


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