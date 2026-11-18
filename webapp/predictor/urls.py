from django.urls import path

from . import views

app_name = "predictor"

urlpatterns = [
    path("", views.predict_view, name="predict"),
    path("batch/", views.batch_view, name="batch"),
    path("rankings/", views.rankings_view, name="rankings"),
    path("insights/", views.insights_view, name="insights"),
    path("api/predict/", views.api_predict, name="api_predict"),
]
