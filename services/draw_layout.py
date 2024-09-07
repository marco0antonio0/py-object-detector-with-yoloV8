import cv2
import numpy as np

def draw_layout(frame, boxes, labels, confs, box_padding=5, label_padding=10):
    """
    Função para desenhar caixas delimitadoras e rótulos no frame, com padding opcional.
    A cor da caixa e do fundo do rótulo é baseada na confiança (quanto maior a confiança, mais verde; quanto menor, mais vermelho).
    
    :param frame: Frame da imagem onde serão desenhados os retângulos e rótulos.
    :param boxes: Lista de coordenadas das caixas delimitadoras.
    :param labels: Lista de rótulos a serem exibidos para cada caixa.
    :param confs: Lista de níveis de confiança para as detecções.
    :param box_padding: Padding aplicado à caixa ao redor do objeto.
    :param label_padding: Padding aplicado ao fundo do rótulo.
    """
    # Garantir que o frame é editável
    if not frame.flags.writeable:
        frame = np.array(frame, copy=True)

    for box, label, conf in zip(boxes, labels, confs):
        x1, y1, x2, y2 = map(int, box)  # Coordenadas do retângulo
        
        # Aplicar o box_padding à caixa, garantindo que as coordenadas fiquem dentro da imagem
        x1 = max(x1 - box_padding, 0)
        y1 = max(y1 - box_padding, 0)
        x2 = min(x2 + box_padding, frame.shape[1])
        y2 = min(y2 + box_padding, frame.shape[0])
        
        # Mapear a confiança (conf) para uma cor de vermelho a verde
        color = get_color_based_on_confidence(conf)

        # Desenhar o retângulo com a cor baseada na confiança
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Obter o tamanho do texto do rótulo
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_width, label_height = label_size

        # Adicionar label_padding ao redor do texto no fundo do rótulo
        label_background_top_left = (x1, y1 - label_height - label_padding)
        label_background_bottom_right = (x1 + label_width + 2 * label_padding, y1)

        # Desenhar um retângulo como fundo do rótulo
        cv2.rectangle(frame, label_background_top_left, label_background_bottom_right, color, -1)

        # Calcular a posição do texto para centralizá-lo no fundo do rótulo
        text_x = x1 + label_padding
        text_y = y1 - (label_padding - label_height // 2)

        # Desenhar o rótulo com texto preto, centralizado no fundo
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def get_color_based_on_confidence(conf):
    """
    Função para mapear um nível de confiança para uma cor que varia de vermelho (baixa confiança) a verde (alta confiança).
    
    :param conf: Nível de confiança (entre 0 e 1).
    :return: Tupla com a cor BGR.
    """
    # Limitar a confiança para o intervalo [0, 1]
    conf = max(0, min(1, conf))
    
    # Interpolação linear entre vermelho (0, 0, 255) e verde (0, 255, 0)
    red = int((1 - conf) * 255)
    green = int(conf * 255)
    
    return (0, green, red)  # Formato BGR (azul, verde, vermelho)
