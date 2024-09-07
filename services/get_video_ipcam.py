import subprocess
from rich.console import Console

# Criar uma instância do rich Console
console = Console()

def get_video_ipcam(rtsp_url, bufsize=6):
    """
    Inicia a captura de frames via FFmpeg a partir de uma URL RTSP, com logs estilizados.
    
    :param rtsp_url: URL RTSP da câmera IP.
    :param bufsize: Tamanho do buffer para a captura de vídeo.
    :return: Subprocesso de captura de vídeo rodando o FFmpeg.
    """
    console.print("✅ [cyan]Iniciada captura de frame com FFmpeg...[/cyan]")
    
    command = [
        'ffmpeg',
        '-i', rtsp_url,  # URL do stream RTSP
        '-f', 'rawvideo',  # Formato de saída
        '-pix_fmt', 'bgr24',  # Formato de pixel para OpenCV
        '-'  # Saída padrão (pipe)
    ]
    
    # Iniciar o subprocesso com o FFmpeg
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**bufsize)
    
    console.print("✅ [bold green]Captura de vídeo iniciada com sucesso![/bold green]")
    return process
