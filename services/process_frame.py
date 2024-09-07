import cv2
from ultralytics import YOLO
import time
import numpy as np

def process_frame(frame, model, custom_labels, conf_threshold=0.20):
    """
    Processa um frame para detectar objetos usando o modelo YOLO.
    
    :param frame: Frame atual do vídeo.
    :param model: Instância do modelo YOLO.
    :param custom_labels: Lista de rótulos personalizados.
    :param conf_threshold: Limite de confiança para as detecções.
    :return: Listas de caixas delimitadoras, rótulos e confidências.
    """
    results = model.predict(frame, conf=conf_threshold,verbose=False)
    boxes = []
    labels = []
    confs = []
    
    for result in results:
        detected_boxes = result.boxes.xyxy.cpu().numpy()
        detected_confs = result.boxes.conf.cpu().numpy()
        detected_classes = result.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(detected_boxes, detected_confs, detected_classes):
            if int(cls) < len(custom_labels):
                label = f"{custom_labels[int(cls)]} {conf:.2f}"
            else:
                label = f"Desconhecido {conf:.2f}"
                
            boxes.append(box)
            labels.append(label)
            confs.append(conf)
    
    return boxes, labels, confs