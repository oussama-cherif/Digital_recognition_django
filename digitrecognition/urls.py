from django.urls import path
from . import views
from .views import predict_digit


urlpatterns = [
    path('', views.upload_and_predict_digit, name='upload_and_predict'),
    path('predict_digit/', predict_digit, name='predict_digit'),
]
