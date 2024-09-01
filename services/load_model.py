import cv2
from ultralytics import YOLO
import time
import numpy as np

def load_model(model_path):
    """
    Carrega o modelo YOLO treinado.
    
    :param model_path: Caminho para o modelo treinado.
    :return: Instância do modelo YOLO carregado.
    """
    return YOLO(model_path)