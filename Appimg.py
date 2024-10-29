import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import dlib
import mediapipe as mp
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Dropout
from tensorflow.keras.models import load_model
from PIL import Image

# Função para criar o modelo facial
def create_face_model(input_shape_image, input_shape_landmarks, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape_image)
    base_model.trainable = True

    image_input = Input(shape=input_shape_image)
    landmarks_input = Input(shape=input_shape_landmarks)

    x = base_model(image_input)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    combined = Concatenate()([x, landmarks_input])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = tf.keras.Model(inputs=[image_input, landmarks_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Inicializações de modelos e MediaPipe
num_classes = 4  # 'Feliz', 'Raiva', 'Surpresa', 'Tristeza'
face_model = create_face_model((224, 224, 3), (68 * 2,), num_classes)
face_model.load_weights('modelo_emocao_face_4classes.weights.h5')

gesture_model = load_model('modelo_landmarks_gesto_emocoes_libras.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Funções de detecção e predição
def detect_face_and_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_region = image[y:y+h, x:x+w]
        rect = dlib.rectangle(x, y, x+w, y+h)
        landmarks = predictor(gray, rect)
        landmark_points = []
        for i in range(68):
            landmark_points.append([landmarks.part(i).x, landmarks.part(i).y])
        return cv2.resize(face_region, (224, 224)), np.array(landmark_points).flatten()
    return np.zeros((224, 224, 3)), np.zeros(68 * 2)

def extract_hand_landmarks(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmark_points = []
        for lm in hand_landmarks.landmark:
            landmark_points.extend([lm.x, lm.y, lm.z])
        return np.array(landmark_points)
    return None

def combine_predictions(face_image, face_landmarks, hand_landmarks):
    face_probs = face_model.predict([np.expand_dims(face_image, axis=0), np.expand_dims(face_landmarks, axis=0)])
    gesture_probs = gesture_model.predict(np.expand_dims(hand_landmarks, axis=0))
    combined_probs = (face_probs + gesture_probs) / 2.0
    final_class = np.argmax(combined_probs, axis=1)
    return final_class, combined_probs

# Interface no Streamlit
st.title("Classificação de Emoções com Detecção Facial e Gestual")

# Upload de imagem
uploaded_file = st.file_uploader("Escolha uma imagem para carregar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Converte para imagem OpenCV
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    original_image = image.copy()

    # Processa a imagem para extração de landmarks
    face_image_resized, face_landmarks = detect_face_and_landmarks(image)
    hand_landmarks = extract_hand_landmarks(image)

    # Classes de emoções
    class_names = ['Feliz', 'Raiva', 'Surpreso', 'Triste']

    if hand_landmarks is not None and face_landmarks is not None:
        # Predição
        emotion_class, probabilities = combine_predictions(face_image_resized, face_landmarks, hand_landmarks)
        
        # Mostra o resultado
        emotion_label = class_names[int(emotion_class)]
        precision = np.max(probabilities)
        text = f"{emotion_label}: {precision * 100:.2f}%"

        # Desenha o texto na imagem
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(original_image, text, (30, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption=text)
    else:
        st.write("Não foi possível detectar a face ou a mão adequadamente.")
