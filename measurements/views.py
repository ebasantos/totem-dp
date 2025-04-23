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
import logging

logger = logging.getLogger(__name__)

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

def glasses_detection(request):
    """View para a página de detecção de óculos."""
    return render(request, 'measurements/glasses_detection.html')

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

    @action(detail=False, methods=['post'])
    def detect_glasses(self, request):
        """
        Detecta se o usuário está usando óculos e calcula a distância
        da pupila até a base da lente.
        """
        try:
            # Obter a imagem da requisição
            if 'image' not in request.data:
                return Response({
                    'success': False,
                    'error': 'Nenhuma imagem fornecida'
                }, status=400)
            
            # Decodificar a imagem
            image_str = request.data['image']
            if ',' in image_str:
                image_data = base64.b64decode(image_str.split(',')[1])
            else:
                image_data = base64.b64decode(image_str)
            
            # Converter para array numpy
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Converter para escala de cinza para detecção
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detectar face
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return Response({
                    'success': False,
                    'error': 'Nenhum rosto detectado'
                }, status=400)
            
            # Pegar a primeira face
            x, y, w, h = faces[0]
            
            # Recortar região dos olhos (aproximadamente)
            roi_y = y + int(h * 0.2)  # 20% do topo da face
            roi_h = int(h * 0.25)     # 25% da altura da face
            roi_x = x
            roi_w = w
            
            eye_region = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Técnica 1: Detecção de óculos usando o classificador específico
            glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
            glasses = glasses_cascade.detectMultiScale(eye_region, 1.1, 4)
            
            # Técnica 2: Detecção de olhos
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(eye_region, 1.1, 4)
            
            # Técnica 3: Análise de bordas na região dos olhos
            edges = cv2.Canny(eye_region, 30, 150)
            edge_density = np.sum(edges > 0) / (roi_w * roi_h)
            
            # Combinação das técnicas para decisão final
            has_glasses = False
            confidence = 0
            
            # Se detectou óculos e poucos olhos, provavelmente está usando óculos
            if len(glasses) > 0 and len(eyes) <= 1:
                has_glasses = True
                confidence = min(100, len(glasses) * 30 + 40)  # Base + bônus por detecção
            
            # Se detectou muitos olhos (reflexos) e alta densidade de bordas
            elif len(eyes) > 2 and edge_density > 0.1:
                has_glasses = True
                confidence = min(100, len(eyes) * 20 + 30)  # Base + bônus por reflexos
            
            # Se detectou poucos olhos e alta densidade de bordas
            elif len(eyes) <= 1 and edge_density > 0.15:
                has_glasses = True
                confidence = min(100, edge_density * 500)  # Base na densidade de bordas
            
            # Se não detectou óculos e encontrou 2 olhos claramente
            elif len(glasses) == 0 and len(eyes) == 2 and edge_density < 0.05:
                has_glasses = False
                confidence = 90  # Alta confiança quando detecta 2 olhos sem interferência
            
            # Caso contrário, decisão baseada na densidade de bordas
            else:
                has_glasses = edge_density > 0.1
                confidence = min(100, edge_density * 400)
            
            return Response({
                'success': True,
                'has_glasses': has_glasses,
                'confidence': round(confidence),
                'details': {
                    'glasses_detections': len(glasses),
                    'eyes_detected': len(eyes),
                    'edge_density': round(edge_density * 100, 1)
                }
            })
                
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=500)

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

def detect_glasses(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Método não permitido'}, status=405)
    
    try:
        # Obter a imagem do corpo da requisição
        try:
            data = json.loads(request.body)
            image_data = data.get('image')
            if not image_data:
                return JsonResponse({'success': False, 'error': 'Nenhuma imagem fornecida'}, status=400)
            
            # Remover o prefixo data:image/jpeg;base64, se presente
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decodificar a imagem base64
            try:
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    return JsonResponse({'success': False, 'error': 'Falha ao decodificar a imagem'}, status=400)
                
                # Pré-processamento da imagem
                # 1. Redimensionar para um tamanho padrão
                height, width = img.shape[:2]
                scale = 640 / width
                img = cv2.resize(img, (640, int(height * scale)))
                
                # 2. Converter para escala de cinza
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # 3. Aplicar equalização de histograma para melhorar o contraste
                gray = cv2.equalizeHist(gray)
                
                # 4. Aplicar filtro bilateral para reduzir ruído mantendo bordas
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
                
                # Carregar os classificadores
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
                
                # Detectar face
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) == 0:
                    return JsonResponse({'success': False, 'error': 'Nenhum rosto detectado'}, status=400)
                
                # Pegar a primeira face
                x, y, w, h = faces[0]
                
                # Desenhar retângulo ao redor da face
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Recortar região dos olhos (aproximadamente)
                roi_y = y + int(h * 0.2)  # 20% do topo da face
                roi_h = int(h * 0.25)     # 25% da altura da face
                roi_x = x
                roi_w = w
                
                eye_region = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                
                # 5. Aplicar morfologia para remover pequenos ruídos
                kernel = np.ones((3,3), np.uint8)
                eye_region = cv2.morphologyEx(eye_region, cv2.MORPH_OPEN, kernel)
                
                # Técnica 1: Detecção de óculos usando o classificador específico
                glasses = glasses_cascade.detectMultiScale(eye_region, 1.1, 4)
                
                # Técnica 2: Detecção de olhos
                eyes = eye_cascade.detectMultiScale(eye_region, 1.1, 4)
                
                # Desenhar retângulos ao redor dos olhos detectados
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(img, (roi_x + ex, roi_y + ey), 
                                (roi_x + ex + ew, roi_y + ey + eh), (0, 255, 0), 2)
                
                # Técnica 3: Análise de bordas na região dos olhos
                edges = cv2.Canny(eye_region, 30, 150)
                edge_density = float(np.sum(edges > 0)) / float(roi_w * roi_h)
                
                # Combinação das técnicas para decisão final
                has_glasses = False
                confidence = 0
                
                # Se detectou óculos e poucos olhos, provavelmente está usando óculos
                if len(glasses) > 0 and len(eyes) <= 1:
                    has_glasses = True
                    confidence = min(100, len(glasses) * 30 + 40)  # Base + bônus por detecção
                    # Desenhar retângulos ao redor dos óculos detectados
                    for (gx, gy, gw, gh) in glasses:
                        cv2.rectangle(img, (roi_x + gx, roi_y + gy), 
                                    (roi_x + gx + gw, roi_y + gy + gh), (0, 0, 255), 2)
                
                # Se detectou muitos olhos (reflexos) e alta densidade de bordas
                elif len(eyes) > 2 and edge_density > 0.1:
                    has_glasses = True
                    confidence = min(100, len(eyes) * 20 + 30)  # Base + bônus por reflexos
                
                # Se detectou poucos olhos e alta densidade de bordas
                elif len(eyes) <= 1 and edge_density > 0.15:
                    has_glasses = True
                    confidence = min(100, edge_density * 500)  # Base na densidade de bordas
                
                # Se não detectou óculos e encontrou 2 olhos claramente
                elif len(glasses) == 0 and len(eyes) == 2 and edge_density < 0.05:
                    has_glasses = False
                    confidence = 90  # Alta confiança quando detecta 2 olhos sem interferência
                
                # Caso contrário, decisão baseada na densidade de bordas
                else:
                    has_glasses = edge_density > 0.1
                    confidence = min(100, edge_density * 400)
                
                # Se detectou óculos com confiança suficiente, calcular a distância
                pupil_to_frame_distance = None
                if has_glasses and confidence >= 70 and len(eyes) > 0:
                    try:
                        # Pegar o primeiro olho detectado
                        eye_x, eye_y, eye_w, eye_h = eyes[0]
                        
                        # Calcular o centro da pupila
                        pupil_center_x = roi_x + eye_x + eye_w // 2
                        pupil_center_y = roi_y + eye_y + eye_h // 2
                        
                        # Desenhar ponto no centro da pupila
                        cv2.circle(img, (pupil_center_x, pupil_center_y), 5, (0, 255, 0), -1)
                        
                        # Encontrar a base da armação (aproximadamente)
                        # Procurar bordas horizontais fortes abaixo do olho
                        eye_bottom = roi_y + eye_y + eye_h
                        search_region = gray[eye_bottom:eye_bottom+100, roi_x:roi_x+roi_w]  # Aumentar área de busca
                        search_edges = cv2.Canny(search_region, 30, 150)
                        
                        # Aplicar dilatação para conectar bordas próximas
                        kernel = np.ones((5,5), np.uint8)
                        search_edges = cv2.dilate(search_edges, kernel, iterations=1)
                        
                        # Encontrar a linha mais forte na região
                        lines = cv2.HoughLinesP(search_edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=20)
                        
                        if lines is not None:
                            # Pegar a linha mais horizontal e mais próxima do olho
                            best_line = None
                            best_score = 0
                            for line in lines:
                                x1, y1, x2, y2 = line[0]
                                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                                if 0 <= angle <= 15 or 165 <= angle <= 180:  # Aumentar tolerância do ângulo
                                    # Calcular distância do olho
                                    line_y = eye_bottom + y1
                                    distance_to_eye = abs(line_y - pupil_center_y)
                                    # Score baseado na força da borda e proximidade do olho
                                    score = np.sum(search_edges[y1:y2+1, x1:x2+1]) / (distance_to_eye + 1)
                                    if score > best_score:
                                        best_score = score
                                        best_line = line[0]
                            
                            if best_line is not None:
                                # Calcular a distância em pixels
                                frame_y = eye_bottom + best_line[1]  # y da linha da armação
                                distance_pixels = frame_y - pupil_center_y
                                
                                # Converter para milímetros (assumindo 1mm = 3.78 pixels)
                                pixels_per_mm = 3.78
                                pupil_to_frame_distance = round(distance_pixels / pixels_per_mm, 1)
                                
                                # Desenhar a linha na imagem
                                cv2.line(img, 
                                        (pupil_center_x, pupil_center_y),
                                        (pupil_center_x, frame_y),
                                        (0, 255, 0), 2)
                                
                                # Adicionar texto com a distância
                                cv2.putText(img, f"{pupil_to_frame_distance}mm",
                                          (pupil_center_x - 30, (pupil_center_y + frame_y) // 2),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                # Desenhar a linha da armação
                                cv2.line(img,
                                        (roi_x + best_line[0], eye_bottom + best_line[1]),
                                        (roi_x + best_line[2], eye_bottom + best_line[3]),
                                        (255, 0, 0), 2)
                    except Exception as e:
                        logger.error(f"Erro ao calcular distância: {str(e)}")
                        pupil_to_frame_distance = None
                
                # Adicionar texto com o status da detecção
                status_text = f"Óculos: {'Sim' if has_glasses else 'Não'} ({confidence}%)"
                cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Converter a imagem com as marcações para base64
                _, buffer = cv2.imencode('.jpg', img)
                marked_image = base64.b64encode(buffer).decode('utf-8')
                
                # Garantir que todos os valores são serializáveis
                response_data = {
                    'success': True,
                    'has_glasses': bool(has_glasses),
                    'confidence': int(round(confidence)),
                    'pupil_to_frame_distance': pupil_to_frame_distance,
                    'marked_image': marked_image,
                    'details': {
                        'glasses_detections': int(len(glasses)),
                        'eyes_detected': int(len(eyes)),
                        'edge_density': float(round(edge_density * 100, 1))
                    }
                }
                
                return JsonResponse(response_data)
                
            except cv2.error as e:
                logger.error(f"Erro OpenCV: {str(e)}")
                return JsonResponse({'success': False, 'error': f'Erro ao processar imagem: {str(e)}'}, status=500)
                
        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'error': 'Formato JSON inválido'}, status=400)
            
    except Exception as e:
        logger.error(f"Erro inesperado: {str(e)}")
        return JsonResponse({'success': False, 'error': f'Erro interno do servidor: {str(e)}'}, status=500)
