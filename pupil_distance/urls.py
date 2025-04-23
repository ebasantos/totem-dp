"""
URL configuration for pupil_distance project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from rest_framework import routers
from measurements.views import PupilDistanceMeasurementViewSet, index, measure, history, settings, view_measurement, frames, glasses_detection, detect_glasses

router = routers.DefaultRouter()
router.register(r'measurements', PupilDistanceMeasurementViewSet)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),
    path('api/glasses/detect/', detect_glasses, name='detect_glasses'),
    path('', index, name='index'),
    path('measure/', measure, name='measure'),
    path('history/', history, name='history'),
    path('settings/', settings, name='settings'),
    path('frames/', frames, name='frames'),
    path('measurements/<int:measurement_id>/', view_measurement, name='view_measurement'),
    path('glasses-detection/', glasses_detection, name='glasses_detection'),
]
