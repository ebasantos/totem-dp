from django.contrib import admin
from .models import PupilDistanceMeasurement

@admin.register(PupilDistanceMeasurement)
class PupilDistanceMeasurementAdmin(admin.ModelAdmin):
    list_display = ('id', 'distance', 'confidence', 'is_validated', 'validated_distance', 'created_at')
    list_filter = ('is_validated', 'created_at')
    search_fields = ('id', 'distance', 'validated_distance')
    readonly_fields = ('created_at', 'updated_at')
