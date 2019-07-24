from django.contrib import admin
from django.contrib.auth import views
from django.urls import path, include

from accounts.forms import LoginForm
from employees import views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('admin/', admin.site.urls),

    # Custome module
    path('employees/', include(('employees.urls', 'employees'), namespace='employees')),

    path('', views.index, name='root_url'),
    path('accounts/',include('accounts.urls')),

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
