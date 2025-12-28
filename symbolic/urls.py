from django.urls import path
from . import views

urlpatterns = [
    path('', views.runtime_gui, name='runtime_gui'),
]