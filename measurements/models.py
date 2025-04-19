from django.db import models

# Create your models here.

class PupilDistanceMeasurement(models.Model):
    distance = models.FloatField(verbose_name='Distância Pupilar')
    confidence = models.FloatField(verbose_name='Nível de Confiança', default=0.0)
    is_validated = models.BooleanField(verbose_name='Validado', default=False)
    validated_distance = models.FloatField(verbose_name='Distância Validada', null=True, blank=True)
    image = models.TextField(verbose_name='Imagem Base64', null=True, blank=True)
    created_at = models.DateTimeField(verbose_name='Data de Criação', auto_now_add=True)
    updated_at = models.DateTimeField(verbose_name='Data de Atualização', auto_now=True)

    class Meta:
        verbose_name = 'Medição de Distância Pupilar'
        verbose_name_plural = 'Medições de Distância Pupilar'
        ordering = ['-created_at']

    def __str__(self):
        return f'Medição {self.id} - {self.distance}mm'
