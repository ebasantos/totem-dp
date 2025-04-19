from rest_framework import serializers
from .models import PupilDistanceMeasurement

class PupilDistanceMeasurementSerializer(serializers.ModelSerializer):
    class Meta:
        model = PupilDistanceMeasurement
        fields = ['id', 'distance', 'confidence', 'is_validated', 'validated_distance', 'created_at', 'updated_at']
        read_only_fields = ['created_at', 'updated_at'] 