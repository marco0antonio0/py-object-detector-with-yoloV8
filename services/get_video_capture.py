import cv2
from ultralytics import YOLO
import time
import numpy as np

def get_video_capture(source=0):
    """
    Configura a captura de vídeo.
    
    :param source: Fonte de vídeo, padrão é '0' para webcam. Pode ser o caminho de um arquivo de vídeo ou URL de uma câmera IP.
    :return: Objeto de captura de vídeo.
    """
    return cv2.VideoCapture(source)