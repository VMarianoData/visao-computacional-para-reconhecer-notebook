import cv2

# Caminho para o arquivo de vídeo
video_path = 'Video.mp4'

# Inicializa a captura de vídeo a partir do arquivo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erro ao abrir o arquivo de vídeo")
else:
    print("Arquivo de vídeo aberto com sucesso")

# Define o tamanho da janela
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 800, 600)  # Define o tamanho desejado da janela

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível ler o frame do vídeo")
        break
    
    # Mostra o frame na janela
    cv2.imshow('Video', frame)
    
    # Aguarda pela tecla 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
