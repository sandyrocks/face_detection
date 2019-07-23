from django.contrib import admin
from django.urls import path, include
from accounts.views import add_accounts

urlpatterns = [
    path('add', add_accounts)
]
