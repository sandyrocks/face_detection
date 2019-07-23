from django.shortcuts import render

# Create your views here.
def add_accounts(request):
    return render(request, 'accounts/add_user.html')
