from services import draw_layout,get_video_capture,load_model,process_frame,cv2,time

def main():
    # Caminho para o modelo treinado
    model_path = 'runs/custom_project/my_experiment17/weights/best.pt'
    
    # Carregar o modelo treinado
    model = load_model(model_path)
    
    # Criar um mapeamento personalizado das classes
    custom_labels = ['bolinha', 'brinquedo caro', 'cachorro', 'cenora de brinquedo', 'garrafinha de agua', 'humano', 'racao']
    
    # Capturar o vídeo da webcam ou da câmera IP
    video_capture = get_video_capture(0)  # Para câmera IP use 'http://<IP>:<PORT>/video'
    
    # Definir o intervalo mínimo entre as atualizações de frame (em segundos)
    update_interval = 0.1  # Intervalo de 100 ms entre as atualizações
    last_update_time = time.time()

    # Armazenar as detecções anteriores
    previous_boxes = []
    previous_labels = []
    previous_confs = []

    # Loop para processar o vídeo frame por frame
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # Se não conseguir ler o frame, sai do loop

        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            # Processar o frame para detectar objetos
            previous_boxes, previous_labels, previous_confs = process_frame(frame, model, custom_labels)
            last_update_time = current_time

        # Desenhar as caixas e rótulos no frame atual
        draw_layout(frame, previous_boxes, previous_labels)

        # Exibir o frame com as detecções
        cv2.imshow("Detections", frame)

        # Se o usuário pressionar 'q', interrompe o loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera o objeto de captura e fecha as janelas
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()