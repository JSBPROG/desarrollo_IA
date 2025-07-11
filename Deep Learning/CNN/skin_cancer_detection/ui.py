import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO

# Cargar modelo
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('first_try.h5')
        return model
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None

model = load_model()

# Título y descripción
st.title("Detector de Tumores de Piel 🧬")
st.write("Sube una imagen o pega una URL para detectar si hay un tumor maligno o no.")

# Input: subir imagen o pegar URL
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
url_input = st.text_input("O pega una URL de la imagen")

# Función de predicción
def predict(image: Image.Image):
    # Preprocesar
    image = image.resize((64, 64))
    img_array = np.expand_dims(np.array(image), axis=0) / 255.0
    prediction = model.predict(img_array)
    return "NO TUMOR" if prediction[0][0] >= 0.5 else "TUMOR"

# Procesar input
image_to_predict = None

if uploaded_file is not None:
    image_to_predict = Image.open(uploaded_file).convert('RGB')
elif url_input:
    try:
        response = requests.get(url_input)
        response.raise_for_status()
        image_to_predict = Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        st.error(f"Error al descargar la imagen: {e}")

# Si hay imagen, mostrar y predecir
if image_to_predict:
    st.image(image_to_predict, caption="Imagen cargada", use_column_width=True)
    
    if model:
        result = predict(image_to_predict)
        st.subheader(f"✅ Predicción: {result}")
    else:
        st.error("El modelo no está cargado.")
