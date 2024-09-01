import cv2
from ultralytics import YOLO
import time
import numpy as np

def draw_layout(frame, boxes, labels, padding=5):
    """
    Função para desenhar caixas delimitadoras e rótulos no frame, com padding opcional.

    :param frame: Frame da imagem onde serão desenhados os retângulos e rótulos.
    :param boxes: Lista de coordenadas das caixas delimitadoras.
    :param labels: Lista de rótulos a serem exibidos para cada caixa.
    :param padding: Quantidade de padding (em pixels) a ser aplicada ao redor das caixas.
    """
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)  # Coordenadas do retângulo
        
        # Aplicar o padding, garantindo que as coordenadas fiquem dentro da imagem
        x1 = max(x1 - padding, 0)
        y1 = max(y1 - padding, 0)
        x2 = min(x2 + padding, frame.shape[1])
        y2 = min(y2 + padding, frame.shape[0])
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Retângulo verde
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Exemplo de uso na função principal ou onde for necessário:
# draw_layout(frame, previous_boxes, previous_labels, padding=10)
