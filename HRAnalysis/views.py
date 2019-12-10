from django.shortcuts import render


def index(request):
    return render(request, 'index.html', {})


def about(request):
    return render(request, 'about.html', {})


def project_members(request):
    return render(request, 'project_members.html', {})


def model_compare(request):
    return render(request, 'model_compare.html', {})


def model_detail(request):
    return render(request, 'model_detail.html', {})


def predict_data(request):
    return render(request, 'predict_data.html', {})


def contact(request):
    return render(request, 'contact.html', {})