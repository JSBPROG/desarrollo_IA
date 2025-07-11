# app.py
import os
import streamlit as st
import requests
from PIL import Image
import io
import base64

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Detector de Fracturas Óseas", layout="centered")

# La URL de la API se obtiene de una variable de entorno para mayor flexibilidad.
# Si no se define, usa el valor por defecto para desarrollo local.
API_URL = os.getenv("API_URL", "http://127.0.0.1:5000/predict")

# --- INTERFAZ DE USUARIO ---
st.title("🔍 Detector de Fracturas en Rayos X")
st.markdown(f"Esta aplicación utiliza un modelo de Deep Learning para predecir si una radiografía contiene una fractura. La API se está ejecutando en: `{API_URL}`")

# Selección del método de entrada
input_method = st.radio(
    "Elige cómo proporcionar la imagen:",
    ("📤 Subir archivo", "🌐 Usar URL"),
    horizontal=True
)

image_to_process = None
payload = None
files = None

# Lógica para cada método de entrada
if input_method == "📤 Subir archivo":
    uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_to_process = uploaded_file
        files = {'file': uploaded_file.getvalue()}

elif input_method == "🌐 Usar URL":
    url = st.text_input("Pega la URL de la imagen aquí:", "https://i.imgur.com/i6S2Y3q.jpeg")
    if url:
        image_to_process = url
        payload = {'url': url}

# Mostrar la imagen cargada
if image_to_process:
    st.image(image_to_process, caption="Imagen a analizar", use_column_width=True)

# Botón para iniciar la predicción
if st.button("🔎 Analizar Imagen", type="primary"):
    if not image_to_process:
        st.warning("Por favor, proporciona una imagen antes de analizar.")
    else:
        with st.spinner("El modelo está pensando... 🧠"):
            try:
                # Enviar la petición a la API
                if files:
                    response = requests.post(API_URL, files=files, timeout=20)
                else:
                    response = requests.post(API_URL, data=payload, timeout=20)
                
                response.raise_for_status()  # Lanza un error para respuestas 4xx/5xx
                
                # --- Mostrar resultados ---
                result = response.json()
                label = result.get("prediction", "desconocido").replace("_", " ").title()
                confidence = result.get("confidence_percent", 0)

                if 'not' in label.lower():
                    st.success(f"**Resultado: {label}** (Confianza: {confidence:.2f}%)")
                else:
                    st.error(f"**Resultado: {label}** (Confianza: {confidence:.2f}%)")

                # Mostrar la imagen con Grad-CAM si está disponible
                if "annotated_image" in result:
                    st.markdown("---")
                    st.markdown("#### 🔥 Visualización del Foco de Atención (Grad-CAM)")
                    st.info("El área resaltada es donde el modelo se concentró para hacer su predicción.")
                    
                    annotated_image_data = result["annotated_image"]
                    st.image(annotated_image_data, caption="Imagen con Grad-CAM", use_column_width=True)

            except requests.exceptions.RequestException as e:
                st.error(f"Error de conexión con la API en {API_URL}. ¿Está el servidor Flask en marcha?")
                st.error(f"Detalles: {e}")
            except Exception as e:
                st.error(f"Ocurrió un error inesperado: {e}")
                # Intentar mostrar el error de la API si está disponible
                try:
                    error_details = response.json().get('error', response.text)
                    st.error(f"Detalles del error de la API: {error_details}")
                except:
                    pass