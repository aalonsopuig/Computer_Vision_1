#!/usr/bin/env python3
"""
Detección de caras con OpenCV y Haar cascade

Este script utiliza la librería OpenCV para realizar detección facial en tiempo real 
mediante el clasificador Haar de detección de rostros frontales. Accede a la cámara 
del ordenador (webcam), procesa los fotogramas de vídeo en escala de grises y 
aplica el algoritmo de detección Haar Cascade incluido en OpenCV. 

En cada iteración, identifica la cara más grande presente en la imagen, dibuja un 
rectángulo azul alrededor de ella y calcula su posición relativa respecto al centro 
del encuadre, mostrando dichas coordenadas en la propia ventana de vídeo. 

El objetivo es ilustrar de forma sencilla el funcionamiento de los clasificadores Haar 
y de las operaciones básicas de captura y visualización de vídeo en OpenCV, sirviendo 
como punto de partida para sistemas más avanzados de visión artificial y robótica. 

Autor: Alejandro Alonso Puig + ChatGPT 4.1
Licencia: Apache 2.0
Julio 2025
"""

import cv2  # Importamos la librería OpenCV para visión artificial

# Ruta del clasificador Haar de detección de caras frontal
haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'      # Localizamos el archivo XML del clasificador
face_cascade = cv2.CascadeClassifier(haar_path)                                # Cargamos el clasificador Haar preentrenado

# Abrimos la cámara (por defecto, dispositivo 0 suele ser la webcam integrada)
cap = cv2.VideoCapture(0)                                                      # Iniciamos la captura de vídeo

if not cap.isOpened():                                                         # Verificamos que la cámara se ha abierto correctamente
    raise RuntimeError("Camera not accesible")                          # Lanzamos error si falla

# Nombre exacto de la ventana (con acento), usado para comprobar si se ha cerrado
ventana = "Face detection"

# Bucle principal de captura y detección
while True:
    ret, frame = cap.read()                                                    # Leemos un frame de vídeo de la cámara
    if not ret:                                                                # Si no se pudo leer, salimos del bucle
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                             # Convertimos el frame a escala de grises (mejora rendimiento)

    faces = face_cascade.detectMultiScale(                                     # Aplicamos el clasificador Haar a la imagen en escala de grises
        gray, scaleFactor=1.1, minNeighbors=5
    )

    h, w = gray.shape                                                          # Obtenemos altura y anchura del frame
    msg_x, msg_y, msg_z = 0.0, 0.0, 0.0                                        # Inicializamos valores de posición relativa

    if len(faces) > 0:                                                         # Si se ha detectado al menos una cara
        x, y, fw, fh = max(faces, key=lambda rect: rect[2] * rect[3])          # Elegimos la más grande (mayor área)
        cx, cy = x + fw // 2, y + fh // 2                                      # Calculamos coordenadas del centro de la cara

        msg_x = (cx - w // 2) / (w // 2)                                       # Coordenada X relativa normalizada [-1, 1]
        msg_y = (cy - h // 2) / (h // 2)                                       # Coordenada Y relativa normalizada [-1, 1]
        msg_z = fw * fh                                                        # Área de la cara como tamaño relativo

        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (255, 0, 0), 2)         # Dibujamos un rectángulo sobre la cara
        cv2.putText(                                                           # Mostramos la posición relativa en texto sobre el frame
            frame,
            f"Rel Pos: ({msg_x:.2f}, {msg_y:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow(ventana, frame)                                                # Mostramos el frame con título "Detección de caras"
    _ = cv2.waitKey(1)                                                         # Refresca la ventana y espera una tecla (sin capturarla)

    # Detectamos si el usuario ha cerrado la ventana manualmente (clic en la X)
    if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
        break                                                                  # Si la ventana ya no está visible, salimos del bucle


# Liberamos recursos al salir
cap.release()                                                                  # Cerramos el acceso a la cámara
cv2.destroyAllWindows()                                                        # Cerramos todas las ventanas de OpenCV
