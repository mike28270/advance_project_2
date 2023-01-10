from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views
from .Dash_Apps import importapp
 
urlpatterns = [
    path('image_upload', views.hotel_image_view, name='image_upload'),
    path('success', views.success, name='success'),
    path('index', views.index, name='index'),
    path('import', views.importapp, name='import')
]
 
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)