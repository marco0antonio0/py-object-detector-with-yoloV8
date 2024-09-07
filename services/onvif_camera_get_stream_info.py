from onvif import ONVIFCamera
from rich.console import Console
from rich.progress import Progress

# Criar uma instância do rich Console
console = Console()

def onvif_camera_get_stream_info(ip, port, username, password, quality='2'):
    """
    Obtém informações do stream de uma câmera IP via ONVIF, com logs estilizados.
    
    :param ip: Endereço IP da câmera.
    :param port: Porta da câmera.
    :param username: Nome de usuário para autenticação ONVIF.
    :param password: Senha para autenticação ONVIF.
    :param quality: Seleção de qualidade ('1' para alta, '2' para baixa).
    :return: URI RTSP, largura do frame, altura do frame, tamanho do frame.
    """
    console.print("⏳ [cyan]Solicitando dados via Onvif...[/cyan]")
    try:
        # Inicializar a câmera
        camera = ONVIFCamera(ip, port, username, password)

        # Obter o serviço de mídia
        media_service = camera.create_media_service()

        # Obter o perfil de vídeo
        profiles = media_service.GetProfiles()
        console.print("✅ [green]Dados recebidos via Onvif[/green]")
        
        if profiles:
            console.print(f"✅ [green]Perfis de vídeo disponíveis: {len(profiles)}[/green]")

            # Exibir as resoluções disponíveis para cada perfil
            for i, profile in enumerate(profiles):
                video_encoder_config = profile.VideoEncoderConfiguration
                if video_encoder_config and hasattr(video_encoder_config, 'Resolution'):
                    frame_width = video_encoder_config.Resolution.Width
                    frame_height = video_encoder_config.Resolution.Height
                    console.print(f"Perfil {i + 1}: Resolução {frame_width}x{frame_height}")
                else:
                    console.print(f"[yellow]Perfil {i + 1}: Resolução não disponível[/yellow]")

            # Escolher o perfil de alta ou baixa qualidade
            if quality == '1':
                profile = max(profiles, key=lambda p: p.VideoEncoderConfiguration.Resolution.Width * p.VideoEncoderConfiguration.Resolution.Height)
            if quality == '2':
                profile = min(profiles, key=lambda p: p.VideoEncoderConfiguration.Resolution.Width * p.VideoEncoderConfiguration.Resolution.Height)
                
            console.print(f"✅ [blue]Perfil {quality} selecionado[/blue]")
            
            # Obter a URI RTSP do perfil selecionado
            stream_uri = media_service.GetStreamUri({
                'StreamSetup': {
                    'Stream': 'RTP-Unicast',
                    'Transport': {'Protocol': 'RTSP'}
                },
                'ProfileToken': profile.token
            })

            # Obter a configuração de codificação de vídeo do perfil selecionado
            video_encoder_config = profile.VideoEncoderConfiguration

            if video_encoder_config and hasattr(video_encoder_config, 'Resolution'):
                frame_width = video_encoder_config.Resolution.Width
                frame_height = video_encoder_config.Resolution.Height
                frame_size = frame_width * frame_height * 3  # Multiplicado por 3 para canais BGR
            else:
                console.print("[red]❌ Resolução não disponível na configuração de codificação de vídeo.[/red]")
                return None, None, None, None

            # Montar a URI completa com autenticação
            rtsp_uri_suffix = stream_uri['Uri'].split("//")[1]
            full_rtsp_uri = f"rtsp://{username}:{password}@{rtsp_uri_suffix}"

            return full_rtsp_uri, frame_width, frame_height, frame_size
        else:
            console.print("[red]❌ Nenhum perfil de vídeo encontrado.[/red]")
            return None, None, None, None
    except Exception as e:
        console.print(f"[red]❌ Erro ao conectar-se à câmera: {e}[/red]")
        return None, None, None, None
