from ultralytics import YOLO
from rich.console import Console


# Criar uma instância do rich Console
console = Console()

def load_model(model_path):
    """
    Carrega o modelo YOLO treinado com feedback visual estilizado.
    
    :param model_path: Caminho para o modelo treinado.
    :param logs: Define se as mensagens do YOLO devem ser exibidas (verbose). Default: False
    :return: Instância do modelo YOLO carregado.
    """

    console.print("⏳ [cyan]Carregando o modelo YOLO...[/cyan]")
    
    # Carregar o modelo YOLO com a opção de logs
    model = YOLO(model=model_path)
    
    console.print("✅ [bold green]Modelo YOLO carregado com sucesso![/bold green]")
    return model
