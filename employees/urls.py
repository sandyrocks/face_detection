from django.contrib.auth import views
from django.urls import path, include

from employees import views

urlpatterns = [
	path('add/', views.add_employee, name='add_employee'),
	path('show/<int:id>', views.show_employee, name='show_employee'),
	path('', views.list_employee, name='list_employee')
]