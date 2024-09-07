from services import draw_layout, load_model, process_frame, onvif_camera_get_stream_info, get_video_ipcam
import time
import cv2
import numpy as np
from rich.console import Console
import os
# Criar uma instância do rich Console
console = Console()

def main():
    console.print("✅ [green]Iniciada programa de detecção por modelo[/green]")
    
    # Dados para conexão
    ip = '192.168.1.40'
    port = 5000
    username = 'admin'
    password = '30ad2010'

    rtsp_url, frame_width, frame_height, frame_size = onvif_camera_get_stream_info(ip, port, username, password)

    process = get_video_ipcam(rtsp_url, 6)

    # Caminho para o modelo treinado
    model_path = 'runs/custom_project/my_experiment17/weights/best.pt'
    
    # Carregar o modelo treinado
    model = load_model(model_path=model_path)
    
    # Criar um mapeamento personalizado das classes
    custom_labels = ['bolinha', 'brinquedo caro', 'cachorro', 'cenora de brinquedo', 'garrafinha de agua', 'humano', 'racao']
    
    # Definir o intervalo mínimo entre as atualizações de frame (em segundos)
    update_interval = 0.2  # Intervalo de 100 ms entre as atualizações
    last_update_time = time.time()

    # Armazenar as detecções anteriores
    previous_boxes = []
    previous_labels = []
    previous_confs = []
    
    console.print("⏳ [cyan]aguardando detecção de vídeo[/cyan]")
    
    # Loop para processar o vídeo frame por frame
    only_pass = False
    while True:
        if not only_pass:
            console.print("✅ [bold green]Iniciada a detecção de vídeo[/bold green]")
            only_pass = True
        # Ler apenas os dados disponíveis no buffer
        if process.stdout.readable():
            raw_frame = process.stdout.read(frame_size)

            # Verifica se o frame está completo
            if len(raw_frame) != frame_size:
                console.print("[red]❌ Falha ao receber frame do stream ou stream terminado[/red]")
                break

            # Converte o frame para um array numpy
            frame = np.frombuffer(raw_frame, np.uint8).reshape((frame_height, frame_width, 3))
            
            # Criar uma cópia do frame para ser editável
            editable_frame = frame.copy()

            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                # Processar o frame para detectar objetos
                previous_boxes, previous_labels, previous_confs = process_frame(editable_frame, model, custom_labels)
                last_update_time = current_time

            # Desenhar as caixas, rótulos e ajustar as cores com base na confiança
            draw_layout(editable_frame, previous_boxes, previous_labels, previous_confs)

            # Exibir o frame com as detecções
            cv2.imshow("Detections", editable_frame)

            # Se o usuário pressionar 'q', interrompe o loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                

    console.print("✅ [green]Detecção de vídeo terminada[/green]")
    
    # Libera o objeto de captura e fecha as janelas
    process.terminate()
    process.wait()  # Aguarda o término do processo ffmpeg
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
