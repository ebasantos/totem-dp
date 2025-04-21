import cv2
import numpy as np
import base64
from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import PupilDistanceMeasurement
from .serializers import PupilDistanceMeasurementSerializer
import os
from django.views.generic import TemplateView
from django.views.decorators.csrf import csrf_exempt
import json
from django.views.decorators.http import require_http_methods

# Create your views here.

def index(request):
    return render(request, 'measurements/index.html')

def measure(request):
    return render(request, 'measurements/measure.html')

def history(request):
    return render(request, 'measurements/history.html')

def settings(request):
    return render(request, 'measurements/settings.html')

def view_measurement(request, measurement_id):
    return render(request, 'measurements/view_measurement.html')

def frames(request):
    return render(request, 'measurements/frames.html')

class PupilDistanceMeasurementViewSet(viewsets.ModelViewSet):
    queryset = PupilDistanceMeasurement.objects.all()
    serializer_class = PupilDistanceMeasurementSerializer

    # Adiciona lista para armazenar as últimas medições
    last_distances = []
    MAX_DISTANCES = 5  # Número de medições para a média móvel

    def get_queryset(self):
        queryset = PupilDistanceMeasurement.objects.all()
        
        # Filtro por data
        date = self.request.query_params.get('date', None)
        if date:
            queryset = queryset.filter(created_at__date=date)
        
        # Filtro por status
        status = self.request.query_params.get('status', None)
        if status == 'validated':
            queryset = queryset.filter(is_validated=True)
        elif status == 'pending':
            queryset = queryset.filter(is_validated=False)
        
        return queryset.order_by('-created_at')

    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['post'])
    def check_position(self, request):
        try:
            # Tenta decodificar o corpo da requisição como JSON
            try:
                data = json.loads(request.body)
                image_data = data.get('image')
            except json.JSONDecodeError:
                # Se falhar, tenta ler como form-data
                image_data = request.POST.get('image')
            
            if not image_data:
                return Response({
                    'face_detected': False,
                    'is_centered': False,
                    'message': 'Nenhuma imagem fornecida'
                })
            
            # Decodifica a imagem base64
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Converte para escala de cinza
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Carrega o classificador de faces
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Ajusta os parâmetros para melhor detecção e estabilidade
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,  # Reduzido para maior estabilidade
                minNeighbors=5,    # Aumentado para maior confiança
                minSize=(100, 100)
            )
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_center_x = x + w/2
                face_center_y = y + h/2
                
                # Verifica se o rosto está centralizado
                image_center_x = image.shape[1] / 2
                image_center_y = image.shape[0] / 2
                
                x_offset = abs(face_center_x - image_center_x)
                y_offset = abs(face_center_y - image_center_y)
                
                # Define uma tolerância menor para centralização
                tolerance = 50  # pixels
                is_centered = x_offset < tolerance and y_offset < tolerance
                
                # Calcula a distância aproximada em milímetros
                pixels_per_mm = 0.264583333
                current_distance = w * pixels_per_mm
                
                # Adiciona a distância atual à lista
                self.last_distances.append(current_distance)
                if len(self.last_distances) > self.MAX_DISTANCES:
                    self.last_distances.pop(0)
                
                # Calcula a média móvel
                distance_mm = round(sum(self.last_distances) / len(self.last_distances), 1)
                
                # Define o intervalo permitido de distância
                min_distance = 78.0  # mm
                max_distance = 80.5  # mm
                target_distance = (min_distance + max_distance) / 2  # média do intervalo desejado
                tolerance = 2.0  # tolerância aumentada para 2mm
                
                # Verifica se está dentro do intervalo com tolerância
                is_distance_ok = abs(distance_mm - target_distance) <= tolerance
                
                if is_centered and is_distance_ok:
                    return Response({
                        'face_detected': True,
                        'is_centered': True,
                        'message': 'Rosto posicionado corretamente',
                        'distance': distance_mm
                    })
                elif not is_centered:
                    return Response({
                        'face_detected': True,
                        'is_centered': False,
                        'message': 'Por favor, centralize seu rosto na tela',
                        'distance': distance_mm
                    })
                else:
                    message = 'Por favor, '
                    if distance_mm < target_distance - tolerance:
                        message += 'aproxime seu rosto da câmera'
                    else:
                        message += 'afaste seu rosto da câmera'
                    
                    return Response({
                        'face_detected': True,
                        'is_centered': False,
                        'message': message,
                        'distance': distance_mm
                    })
            else:
                # Limpa a lista de distâncias quando não há face detectada
                self.last_distances = []
                return Response({
                    'face_detected': False,
                    'is_centered': False,
                    'message': 'Nenhum rosto detectado. Por favor, certifique-se de que seu rosto está visível e bem iluminado.'
                })
                
        except Exception as e:
            print(f"Erro ao processar imagem: {str(e)}")
            return Response({
                'face_detected': False,
                'is_centered': False,
                'message': f'Erro ao processar imagem: {str(e)}'
            })

    @action(detail=False, methods=['post'])
    def measure(self, request):
        try:
            image_data = request.data.get('image')
            if not image_data:
                return Response({'error': 'Nenhuma imagem fornecida'}, status=400)

            # Remove o prefixo da string base64 se presente
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            # Decodifica a imagem base64
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Carrega os classificadores
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

            # Converte para escala de cinza
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detecta faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return Response({'error': 'Nenhum rosto detectado'}, status=400)

            # Pega a primeira face detectada
            (x, y, w, h) = faces[0]

            # Verifica se a face está centralizada
            height, width = image.shape[:2]
            center_x = width // 2
            center_y = height // 2
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # Define uma tolerância para centralização
            tolerance_center = 50  # pixels
            if abs(face_center_x - center_x) > tolerance_center or abs(face_center_y - center_y) > tolerance_center:
                return Response({'error': 'Rosto não está centralizado'}, status=400)

            # Detecta olhos na região da face
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
            
            if len(eyes) < 2:
                return Response({'error': 'Menos de 2 olhos detectados'}, status=400)

            # Ordena os olhos da esquerda para a direita
            eyes = sorted(eyes, key=lambda e: e[0])
            
            # Pega os dois olhos mais centrais
            if len(eyes) > 2:
                # Calcula o centro da face
                face_center = w // 2
                # Encontra os dois olhos mais próximos do centro
                eyes = sorted(eyes, key=lambda e: abs((e[0] + e[2]//2) - face_center))[:2]

            # Calcula a distância entre os olhos
            eye1_x = eyes[0][0] + eyes[0][2]//2
            eye2_x = eyes[1][0] + eyes[1][2]//2
            distance_pixels = abs(eye2_x - eye1_x)

            # Define a distância ideal e tolerância
            ideal_distance = 100  # pixels
            tolerance = 10  # pixels
            
            if abs(distance_pixels - ideal_distance) > tolerance:
                return Response({'error': 'Rosto muito próximo ou muito distante'}, status=400)

            # Converte pixels para milímetros
            pixels_per_mm = 0.264583333
            distance_mm = distance_pixels * pixels_per_mm

            # Salva a medição
            measurement = PupilDistanceMeasurement.objects.create(
                distance=distance_mm,
                confidence=0.9,
                image=image_data
            )

            return Response({
                'distance': distance_mm,
                'confidence': 0.9,
                'id': measurement.id
            })

        except Exception as e:
            return Response({'error': str(e)}, status=500)

    @action(detail=True, methods=['post'])
    def validate(self, request, pk=None):
        try:
            measurement = self.get_object()
            validated_distance = request.data.get('validated_distance')
            
            if validated_distance is None:
                return Response({'error': 'Distância validada não fornecida'}, status=400)

            measurement.validated_distance = validated_distance
            measurement.is_validated = True
            measurement.save()

            return Response({
                'id': measurement.id,
                'distance': measurement.distance,
                'validated_distance': measurement.validated_distance,
                'is_validated': measurement.is_validated
            })

        except Exception as e:
            return Response({'error': str(e)}, status=500)

class FramesView(View):
    def get_context_data(self, **kwargs):
        frames = [
            {
                'id': 1,
                'name': 'Óculos 1',
                'image': 'frames/frame1.png',
                'price': '299,90',
                'measurements': {
                    'width': 140,
                    'height': 45,
                    'bridge': 18
                }
            },
            {
                'id': 2,
                'name': 'Óculos 2',
                'image': 'frames/frame2.png',
                'price': '399,90',
                'measurements': {
                    'width': 145,
                    'height': 48,
                    'bridge': 20
                }
            }
        ]
        return {'frames': frames}

    def get(self, request, *args, **kwargs):
        context = self.get_context_data()
        return render(request, 'measurements/frames.html', context)
