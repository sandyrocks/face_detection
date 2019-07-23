from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),

    # Custom urls here
    path('accounts/', include('accounts.urls'))
]
