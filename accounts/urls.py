from django.contrib import admin
from django.contrib.auth import views
from django.urls import path,re_path, include

from accounts.forms import LoginForm

urlpatterns = [
	path('login/',views.LoginView.as_view(authentication_form=LoginForm), name='login'),
	path('logout/', views.LogoutView.as_view(), name='logout')
]
