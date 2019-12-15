from django import forms

from .models import PredictFormData


class PredictForm(forms.Form):
    algortihm = forms.CharField(max_length=200)
    fname = forms.CharField(max_length=200)