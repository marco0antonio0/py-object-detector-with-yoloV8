from ultralytics import YOLO
import matplotlib.pyplot as plt

# Carregar o modelo YOLOv8 pré-treinado
model = YOLO('yolov8n.pt')  # Use o modelo de sua escolha

# Treinar o modelo
results = model.train(data='dataset/data.yaml', epochs=300, imgsz=640, project='runs/custom_project', name='my_experiment', patience=0)

