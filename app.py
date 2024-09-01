import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_annotations(annotation_dir, images_dir, scale_factor=5):
    annotations = []

    for xml_file in os.listdir(annotation_dir):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(annotation_dir, xml_file))
            root = tree.getroot()
            filename = root.find('filename').text
            image_path = os.path.join(images_dir, filename)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Imagem não encontrada ou inválida: {image_path}")
                continue

            # Reduza a imagem pela escala definida
            original_height, original_width = img.shape[:2]
            new_width = original_width // scale_factor
            new_height = original_height // scale_factor
            img = cv2.resize(img, (new_width, new_height))

            objects = []
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                label = obj.find('name').text
                xmin = int(bbox.find('xmin').text) // scale_factor
                ymin = int(bbox.find('ymin').text) // scale_factor
                xmax = int(bbox.find('xmax').text) // scale_factor
                ymax = int(bbox.find('ymax').text) // scale_factor
                objects.append((label, xmin, ymin, xmax, ymax))
            annotations.append((img, objects))
    return annotations, (new_width, new_height)

def prepare_data(annotations, input_size):
    X = []
    y = []

    for img, objects in annotations:
        for label, xmin, ymin, xmax, ymax in objects:
            # Certifique-se de que as coordenadas estão dentro dos limites da imagem
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img.shape[1], xmax)
            ymax = min(img.shape[0], ymax)

            # Extraia a região de interesse
            roi = img[ymin:ymax, xmin:xmax]

            # Verifique se o ROI não está vazio
            if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                print(f"Ignorando ROI vazia ou inválida com coordenadas ({xmin}, {ymin}, {xmax}, {ymax}) na imagem de shape {img.shape}")
                continue

            roi = cv2.resize(roi, input_size)
            X.append(roi)
            y.append(1)  # Considerando 1 para objeto presente, pode-se adaptar para múltiplas classes

    X = np.array(X, dtype=np.float32) / 255.0
    y = to_categorical(np.array(y), num_classes=2)  # Supondo duas classes: objeto presente e ausente
    return X, y

def create_robust_model(input_shape):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(512, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # Saída para 2 classes: objeto presente/ausente
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X, y, input_size):
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Conjunto de dados vazio. Verifique se os dados foram carregados corretamente.")
    
    model = create_robust_model(input_shape=(input_size[1], input_size[0], 3))  # Ajuste para usar o input_size correto
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    return model, history

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def detect_objects(model, image_path, input_size):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao carregar a imagem para detecção: {image_path}")
        return
    img_resized = cv2.resize(img, input_size)
    img_resized = np.expand_dims(img_resized, axis=0) / 255.0
    
    prediction = model.predict(img_resized)
    
    if np.argmax(prediction) == 1:  # Supondo que '1' é a classe do objeto
        print("Object detected!")
        cv2.imshow("Detected Object", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No object detected.")

def main():
    # Defina os diretórios
    annotation_dir = '/home/marco/Documentos/dev/py-object-detector/dataset/annotations'
    images_dir = '/home/marco/Documentos/dev/py-object-detector/dataset/images'

    # Carregue e prepare os dados
    annotations, input_size = load_annotations(annotation_dir, images_dir)
    X, y = prepare_data(annotations, input_size)

    # Verifique se o conjunto de dados não está vazio
    if len(X) == 0 or len(y) == 0:
        print("Nenhum dado válido encontrado para treinamento. Verifique as anotações e imagens.")
        return

    # Treine o modelo
    model, history = train_model(X, y, input_size)

    # Salve o modelo treinado
    model.save('/home/marco/Documentos/dev/py-object-detector/saved_model/object_detection_model.h5')
    print("Modelo salvo com sucesso em '/home/marco/Documentos/dev/py-object-detector/saved_model/object_detection_model.h5'.")

    # Exiba o gráfico de epochs x accuracy
    plot_accuracy(history)

    # Escolha uma imagem aleatória para a detecção
    random_image_path = random.choice([ann[0] for ann in annotations])

    # Faça a detecção de objetos
    detect_objects(model, random_image_path, input_size)

if __name__ == "__main__":
    main()
