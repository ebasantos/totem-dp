from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('measure/', views.measure, name='measure'),
    path('history/', views.history, name='history'),
    path('settings/', views.settings, name='settings'),
    path('measurement/<int:measurement_id>/', views.view_measurement, name='view_measurement'),
    path('frames/', views.frames, name='frames'),
    path('glasses_detection/', views.glasses_detection, name='glasses_detection'),
    path('api/glasses/detect/', views.detect_glasses, name='detect_glasses'),
] 