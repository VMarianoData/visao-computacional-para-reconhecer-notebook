import cv2
from roboflow import Roboflow

# Inicializar a API do Roboflow
rf = Roboflow(api_key="Zw0So7rpFtNMSdZvWwk2")
project = rf.workspace().project("legal-ffa1p")
model = project.version(1).model

# Caminho para o arquivo de vídeo
video_path = 'Video.mp4'

# Inicializa a captura de vídeo a partir do arquivo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erro ao abrir o arquivo de vídeo")
else:
    print("Arquivo de vídeo aberto com sucesso")

# Define o tamanho da janela
cv2.namedWindow('Object Recognition', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Recognition', 800, 600)  # Define o tamanho desejado da janela

# Função para salvar frames temporariamente
def save_temp_frame(frame, filename="temp_frame.jpg"):
    cv2.imwrite(filename, frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível ler o frame do vídeo")
        break
    
    # Redimensionar o frame para um tamanho mais manejável
    frame = cv2.resize(frame, (800, 600))

    # Salvar frame temporariamente
    save_temp_frame(frame)
    
    # Fazer a inferência
    prediction = model.predict("temp_frame.jpg", confidence=40, overlap=30).json()
    
    # Verificar se todos os objetos estão presentes
    objects_present = {'Notebook': False, 'Cadeado': False, 'Carregador': False}
    notebook_coords = None
    carregador_inside_notebook = False
    cadeado_inside_notebook = False
    
    for pred in prediction['predictions']:
        label = pred['class']
        if label in objects_present:
            objects_present[label] = True
        if label == 'Notebook':
            notebook_coords = (pred['x'], pred['y'], pred['width'], pred['height'])
        elif label == 'Carregador':
            # Verificar se o Carregador está dentro ou tocando o Notebook
            if (notebook_coords and
                pred['x'] >= notebook_coords[0] and 
                pred['y'] >= notebook_coords[1] and
                pred['x'] + pred['width'] <= notebook_coords[0] + notebook_coords[2] and 
                pred['y'] + pred['height'] <= notebook_coords[1] + notebook_coords[3]):
                carregador_inside_notebook = True
        elif label == 'Cadeado':
            # Verificar se o Cadeado está dentro ou tocando o Notebook
            if (notebook_coords and
                pred['x'] >= notebook_coords[0] and 
                pred['y'] >= notebook_coords[1] and
                pred['x'] + pred['width'] <= notebook_coords[0] + notebook_coords[2] and 
                pred['y'] + pred['height'] <= notebook_coords[1] + notebook_coords[3]):
                cadeado_inside_notebook = True
    
    # Determinar a cor da bounding box (verde se o Carregador e o Notebook estão juntos, ou o Cadeado e o Notebook estão juntos)
    if carregador_inside_notebook:
        box_color_carregador = (0, 255, 0)  # Verde para Carregador
    else:
        box_color_carregador = (0, 0, 255)  # Vermelho para Carregador
    
    if cadeado_inside_notebook:
        box_color_cadeado = (0, 255, 0)  # Verde para Cadeado
    else:
        box_color_cadeado = (0, 0, 255)  # Vermelho para Cadeado
    
    # Mostrar o resultado
    for pred in prediction['predictions']:
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        label = pred['class']
        confidence = pred['confidence']
        
        # Determinar a cor da bounding box baseada no objeto
        if label == 'Carregador':
            box_color = box_color_carregador
        elif label == 'Cadeado':
            box_color = box_color_cadeado
        else:
            box_color = (0, 0, 255)  # Vermelho para outros objetos
        
        # Desenhar a bounding box e label no frame
        start_point = (int(x - w / 2), int(y - h / 2))
        end_point = (int(x + w / 2), int(y + h / 2))
        cv2.rectangle(frame, start_point, end_point, box_color, 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (start_point[0], start_point[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
    cv2.imshow('Object Recognition', frame)
    
    # Aguarda pela tecla 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
