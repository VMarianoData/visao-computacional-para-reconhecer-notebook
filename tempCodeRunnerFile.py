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
    
    for pred in prediction['predictions']:
        label = pred['class']
        if label in objects_present:
            objects_present[label] = True
        if label == 'Notebook':
            notebook_coords = (pred['x'], pred['y'], pred['width'], pred['height'])
    
    # Verificar a sobreposição do Cadeado e Carregador no Notebook
    if all(objects_present.values()) and notebook_coords:
        touching_cadeado = False
        touching_carregador = False
        
        for pred in prediction['predictions']:
            if pred['class'] == 'Cadeado':
                if (pred['x'] + pred['width'] >= notebook_coords[0] and 
                    pred['x'] <= notebook_coords[0] + notebook_coords[2] and 
                    pred['y'] + pred['height'] >= notebook_coords[1] and 
                    pred['y'] <= notebook_coords[1] + notebook_coords[3]):
                    touching_cadeado = True
            elif pred['class'] == 'Carregador':
                if (pred['x'] + pred['width'] >= notebook_coords[0] and 
                    pred['x'] <= notebook_coords[0] + notebook_coords[2] and 
                    pred['y'] + pred['height'] >= notebook_coords[1] and 
                    pred['y'] <= notebook_coords[1] + notebook_coords[3]):
                    touching_carregador = True
        
        if touching_cadeado and touching_carregador:
            box_color = (0, 255, 0)  # Verde
        else:
            box_color = (0, 0, 255)  # Vermelho
    else:
        box_color = (0, 0, 255)  # Vermelho
    
    # Mostrar o resultado
    for pred in prediction['predictions']:
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        label = pred['class']
        confidence = pred['confidence']
        
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
