# Projeto de Detecção de Objetos com YOLOv8

Este projeto é uma aplicação de detecção de objetos utilizando o modelo YOLOv8 (You Only Look Once) treinado em um conjunto de dados personalizado. O projeto é modularizado para facilitar a leitura e manutenção do código.

## Estrutura do Projeto

- **main.py**: Arquivo principal que executa a detecção de objetos em tempo real usando a webcam ou uma câmera IP.
- **utils.py**: Contém funções utilitárias como `draw_layout`, responsável por desenhar as caixas delimitadoras e rótulos nos frames.
- **runs/**: Pasta onde os modelos treinados e os resultados são armazenados.
- **dataset/**: Contém os dados de treinamento e validação do modelo que foram gerados apartir de um rotulador de imagem, uma dica, uso o roboflow para gerar esse dataset e cole nesta pasta.

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

## Passos a Seguir

### 1. Acesse o Roboflow e Crie um Projeto de Detecção de Objetos

1. Acesse o [Roboflow](https://roboflow.com/) e crie uma conta ou faça login.
2. Crie um novo projeto e selecione "Object Detection" como o tipo de projeto.
3. Dê um nome ao seu projeto e clique em "Create Project".

### 2. Selecione e Anote Imagens

1. Faça upload de pelo menos 5 imagens que você deseja rotular para o treinamento.
2. Clique em "Annotate" para começar a marcar os objetos nas imagens usando caixas delimitadoras ou polígonos.
3. Certifique-se de que todos os objetos nas imagens estão corretamente anotados.

### 3. Baixe o Dataset no Formato YOLOv8

1. Após concluir a anotação, clique em "Generate" no canto superior direito.
2. Escolha "YOLOv8" como o formato de exportação.
3. Baixe o dataset, que incluirá as imagens e um arquivo `data.yaml` necessário para o treinamento.

### 4. Indique o Caminho para `data.yaml` no Script de Treinamento

1. No diretório do seu projeto, abra o script `train.py`.
2. Defina o caminho para o arquivo `data.yaml` dentro do dataset que você baixou.
   ```python
   data = '/caminho/para/seu/dataset/data.yaml'
   ```

### 5. Especifique os Caminhos Corretos para `[train, val, test]`

1. Dentro do arquivo `data.yaml`, certifique-se de que os caminhos para `train`, `val` e `test` estão corretos.
2. Ajuste os caminhos, se necessário:
   ```yaml
   train: /caminho/para/seu/dataset/train
   val: /caminho/para/seu/dataset/val
   test: /caminho/para/seu/dataset/test
   ```

### 6. Execute o Treinamento do Modelo

1. Abra um terminal ou ambiente de desenvolvimento Python.
2. Navegue até o diretório que contém o script `train.py`.
3. Execute o comando de treinamento:
   ```bash
   python train.py --data /caminho/para/seu/dataset/data.yaml --epochs 100 --weights yolov8n.pt
   ```

### 7. Execute o `main.py` para Fazer a Predição em Tempo Real

1. Após o treinamento, crie ou modifique o script `main.py` para realizar predições em tempo real.
2. Carregue o modelo treinado e execute predições em imagens ou vídeos ao vivo:

   ```python
   from ultralytics import YOLO

   model = YOLO('/caminho/para/seu/modelo/treinado.pt')  # Carregue o modelo treinado
   results = model.predict(source='video.mp4')  # Execute a predição em um vídeo ou webcam
   ```

### 8. Instale os Requisitos

1. Certifique-se de que todas as dependências estão instaladas antes de executar os scripts de treinamento e predição.
2. Instale as dependências usando:
   ```bash
   pip install -r requirements.txt
   ```

## Notas

- Certifique-se de que os caminhos estão corretos nos seus scripts.
- Utilize uma GPU para tempos de treinamento mais rápidos.

## Licença

Este projeto está licenciado sob a Licença MIT.

## Resultados

Os resultados das detecções são exibidos em tempo real com caixas verdes ao redor dos objetos identificados, junto com o rótulo correspondente e a confiança da detecção.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir um pull request ou relatar problemas.

## Licença

Este projeto é licenciado sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.
