from ultralytics import YOLO

# Carregar o modelo YOLOv8 pré-treinado
model = YOLO('yolov8n.pt')
# altere conforme sua necessidade
data = "dataset/data.yaml"

# Treinar o modelo
results = model.train(data=data, epochs=300, imgsz=640, project='runs/custom_project', name='my_experiment', patience=0)

