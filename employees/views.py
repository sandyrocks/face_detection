import os 
import glob 

from django.shortcuts import render, redirect,reverse

from employees.forms import EmployeeForm
from employees.models import Employee

# Create your views here.
def index(request):
    return render(request, 'employees/index.html')


def add_employee(request):
    if request.method == "POST":
        form = EmployeeForm(request.POST)
        files = request.FILES.getlist('file_field')

        if form.is_valid():
            employee_obj = form.save()
            for f in files:
                handle_uploaded_file(f, employee_obj)
            return redirect(f"/employees/show/{employee_obj.id}")
        else:
            return render(request, 'employees/add.html', {'form': form })        
    else:
        form = EmployeeForm()
    return render(request, 'employees/add.html', {'form': form })


def show_employee(request, id=None):
    if id is not None:
        employee = Employee.objects.get(id=id)
        path= f"uploads/{employee.username}/"
        files_list = [f for f in glob.glob(path + "**/*", recursive=True)]
        files = []
        for file in files_list:
            _, folder, filename = file.split("/")
            files.append(f"{folder}/{filename}") 
        return render(request, 'employees/show.html', {'employee':employee, 'files': files})


def list_employee(request):
    employees = Employee.objects.all()
    return render(request, 'employees/list.html', {'employees': employees})


def train_employee(request):
    return render(request, 'employees/index.html')


def handle_uploaded_file(f, emp_object):
    os.chdir('uploads')
    if not os.path.isdir(emp_object.username):
        os.mkdir(emp_object.username)
    os.chdir("..")
    employee_storage = f"uploads/{emp_object.username}/{f.name}"
    with open(employee_storage, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)