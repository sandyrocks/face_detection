from django.forms import ModelForm
from employees.models import Employee
from django import forms


class EmployeeForm(ModelForm):
    file_field = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}), required = False)
    class Meta:
        model = Employee
        fields = ['first_name', 'last_name', 'department' , 'username' , 'file_field']