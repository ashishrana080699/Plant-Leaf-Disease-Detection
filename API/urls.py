from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('',views.__index__function),
    path('predict',views.predict_plant_disease)
]