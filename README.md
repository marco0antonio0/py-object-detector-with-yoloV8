
# Projeto de Detecção de Objetos com YOLOv8

Este projeto é uma aplicação de detecção de objetos utilizando o modelo YOLOv8 (You Only Look Once) treinado em um conjunto de dados personalizado. O projeto é modularizado para facilitar a leitura e manutenção do código.

## Estrutura do Projeto

- **main.py**: Arquivo principal que executa a detecção de objetos em tempo real usando a webcam ou uma câmera IP.
- **utils.py**: Contém funções utilitárias como `draw_layout`, responsável por desenhar as caixas delimitadoras e rótulos nos frames.
- **runs/**: Pasta onde os modelos treinados e os resultados são armazenados.
- **dataset/**: Contém os dados de treinamento e validação do modelo.

## Funcionalidades

- **Detecção de Objetos em Tempo Real**: Detecta e classifica objetos em tempo real utilizando a webcam ou uma câmera IP.
- **Modularização**: O código está dividido em módulos para facilitar a reutilização e extensão.
- **Personalização de Rótulos**: Os rótulos das classes podem ser personalizados para refletir os nomes desejados.
- **Padding Ajustável**: É possível definir um padding ao redor das caixas delimitadoras para melhorar a visualização dos objetos detectados.

## Requisitos

- Python 3.8 ou superior
- Bibliotecas: `opencv-python`, `ultralytics`, `numpy`

Você pode instalar as dependências necessárias usando:

```bash
pip install -r requirements.txt
```

## Como Usar

1. **Treinamento do Modelo**:

   Se você deseja treinar um novo modelo, pode usar o comando abaixo. Certifique-se de que o arquivo `dataset.yaml` está configurado corretamente:

   ```python
   from ultralytics import YOLO

   model = YOLO('yolov8n.pt')
   results = model.train(data='dataset/data.yaml', epochs=300, imgsz=640, project='runs/custom_project', name='my_experiment', patience=0)
   ```

2. **Detecção em Tempo Real**:

   Para executar a detecção de objetos em tempo real usando a webcam ou uma câmera IP:

   ```python
   import cv2
   from ultralytics import YOLO
   from utils import draw_layout

   model = YOLO('runs/custom_project/my_experiment17/weights/best.pt')  
   custom_labels = ['bolinha', 'brinquedo caro', 'cachorro', 'cenora de brinquedo', 'garrafinha de agua', 'humano', 'racao']
   video_capture = cv2.VideoCapture(0)  # Ou use a URL da câmera IP

   while True:
       ret, frame = video_capture.read()
       if not ret:
           break

       results = model(frame, conf=0.50)
       boxes = [r.boxes.xyxy.cpu().numpy() for r in results]
       labels = [f"{custom_labels[int(r.boxes.cls.cpu().numpy()[0])]} {r.boxes.conf.cpu().numpy()[0]:.2f}" for r in results]

       draw_layout(frame, boxes, labels, padding=10)
       cv2.imshow("Detections", frame)

       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   video_capture.release()
   cv2.destroyAllWindows()
   ```

3. **Personalização de Rótulos**:

   Se você deseja alterar os rótulos das classes para algo mais específico, edite a lista `custom_labels`:

   ```python
   custom_labels = ['racao', 'bolinha', 'brinquedo caro', 'cenora de brinquedo', 'garrafinha de agua', 'humano', 'cachorro']
   ```

## Resultados

Os resultados das detecções são exibidos em tempo real com caixas verdes ao redor dos objetos identificados, junto com o rótulo correspondente e a confiança da detecção.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir um pull request ou relatar problemas.

## Licença

Este projeto é licenciado sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.
