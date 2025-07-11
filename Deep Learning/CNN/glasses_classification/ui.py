import gradio as gr
from PIL import Image
import numpy as np
import tensorflow as tf
import requests

# Cargar el modelo entrenado
try:
    model = tf.keras.models.load_model('first_try.h5')
except (IOError, ImportError) as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'first_try.h5' is in the correct directory.")
    model = None

def predict_image(input_image):
    """
    Predice si la imagen contiene gafas a partir de una imagen de entrada (URL o subida).
    """
    if model is None:
        return "Error: Modelo no cargado."

    if isinstance(input_image, str):  # Si es una URL
        try:
            response = requests.get(input_image, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw).convert('RGB')
        except requests.exceptions.RequestException as e:
            return f"Error al descargar la imagen: {e}"
    else:  # Si es una imagen subida (numpy array)
        image = Image.fromarray(input_image).convert('RGB')

    # Preprocesar la imagen
    image = image.resize((64, 64))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Realizar la predicción
    prediction = model.predict(img_array)
    
    # Devolver el resultado
    if prediction[0][0] >= 0.5:
        return "SIN GAFAS"
    else:
        return "GAFAS"

# Crear la interfaz de Gradio
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Sube una imagen o pega una URL"),
    outputs=gr.Label(num_top_classes=1, label="Predicción"),
    title="Clasificador de Gafas",
    description="Sube una imagen para detectar si una persona lleva gafas."
)

# Lanzar la interfaz
if __name__ == "__main__":
    if model is not None:
        iface.launch()
    else:
        print("La interfaz no se puede iniciar porque el modelo no se pudo cargar.")
