import os 
import glob 

from django.shortcuts import render, redirect,reverse

from employees.forms import EmployeeForm
from employees.models import Employee

# Create your views here.
def index(request):
    return render(request, 'employees/index.html')


def add_employee(request):
    button_title = "Create Employee"
    if request.method == "POST":
        form = EmployeeForm(request.POST)
        files = request.FILES.getlist('file_field')

        if form.is_valid():
            employee_obj = form.save()
            for f in files:
                handle_uploaded_file(f, employee_obj)
            return redirect(f"/employees/show/{employee_obj.id}")
        else:
            return render(request, 'employees/add.html', {'form': form, 'button_title': button_title })        
    else:
        form = EmployeeForm()
    return render(request, 'employees/add.html', {'form': form, 'button_title': button_title})


def show_employee(request, id=None):
    if id is not None:
        employee = Employee.objects.get(id=id)
        images = get_images(employee)
        return render(request, 'employees/show.html', {'employee':employee, 'images': images})


def list_employee(request):
    employees = Employee.objects.all()
    return render(request, 'employees/list.html', {'employees': employees})


def train_employee(request):
    return render(request, 'employees/index.html')


def edit_employee(request, id=None):
    button_title = "Update Employee"
    if id is not None:
        employee = Employee.objects.get(id=id)
        if request.method == "POST":
            form = EmployeeForm(request.POST, instance=employee)
            files = request.FILES.getlist('file_field')

            if form.is_valid():
                employee_obj = form.save()
                for f in files:
                    handle_uploaded_file(f, employee_obj)
                return redirect(f"/employees/show/{employee_obj.id}")
        else:
            form = EmployeeForm(instance=employee)
            images = get_images(employee)
            return render(request, 'employees/add.html', {'form': form, 'images': images, 'button_title': button_title})

def handle_uploaded_file(f, emp_object):
    os.chdir('uploads')
    if not os.path.isdir(emp_object.username):
        os.mkdir(emp_object.username)
    os.chdir("..")
    employee_storage = f"uploads/{emp_object.username}/{f.name}"
    with open(employee_storage, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def get_images(employee):
    path= f"uploads/{employee.username}/"
    image_list = [f for f in glob.glob(path + "**/*", recursive=True)]
    images = []
    for file in image_list:
        _, folder, filename = file.split("/")
        images.append(f"{folder}/{filename}")
    return images