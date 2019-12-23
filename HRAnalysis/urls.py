from django.urls import path
from HRAnalysis import views

urlpatterns = [
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('project_members/', views.project_members, name='project_members'),
    path('dashboards/model_compare/', views.model_compare, name='model_compare'),
    path('dashboards/model_detail/', views.model_detail, name='model_detail'),
    path('dashboards/data_detail/', views.data_detail, name='data_detail'),
    path('predict_data/', views.predict_data, name='predict_data'),
    path('contact/', views.contact, name='contact'),
]