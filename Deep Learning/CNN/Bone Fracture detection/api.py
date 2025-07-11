# api.py
import os
import json
import base64
import io
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import requests

# --- CONFIGURACIÓN Y CARGA DE MODELO ---
MODEL_PATH = 'bone_fracture_model.keras'
CLASS_INDICES_PATH = 'class_indices.json'

app = Flask(__name__)
model = None
class_indices = None

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Modelo cargado correctamente.")
    else:
        print(f"🚨 ERROR: No se encontró el archivo del modelo en '{MODEL_PATH}'.")

    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
            # Invertir el diccionario para mapear de índice a etiqueta
            class_labels = {v: k for k, v in class_indices.items()}
        print(f"✅ Índices de clase cargados: {class_labels}")
    else:
         print(f"🚨 ERROR: No se encontró el archivo de índices en '{CLASS_INDICES_PATH}'.")

except Exception as e:
    print(f"🚨 Error crítico durante la inicialización: {e}")

def preprocess_image(image, target_size=(224, 224)):
    """Prepara una imagen PIL para la predicción."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    arr = tf.keras.preprocessing.image.img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)

def find_last_conv_layer(model):
    """Encuentra el nombre de la última capa convolucional en el modelo base."""
    # El modelo base (MobileNetV2) es generalmente la segunda capa del modelo funcional.
    base_model = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)), None)
    if not base_model:
        raise ValueError("No se pudo encontrar el modelo base (ej. MobileNetV2) dentro del modelo principal.")
        
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No se encontró ninguna capa convolucional en el modelo base.")

def get_gradcam_heatmap(img_array, last_conv_layer_name):
    """Genera un mapa de calor Grad-CAM."""
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Usamos la predicción de la clase "fractured" (asumiendo que es la clase 0)
        # Si la predicción es > 0.5, el modelo se inclina por la clase 1 ("not fractured")
        # Si la predicción es < 0.5, se inclina por la clase 0 ("fractured")
        # El gradiente debe calcularse con respecto a la clase predicha.
        pred_index = tf.cast(predictions[0][0] > 0.5, tf.int32)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def superimpose_gradcam(original_img_bgr, heatmap, alpha=0.6):
    """Superpone el heatmap sobre la imagen original."""
    h, w = original_img_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(original_img_bgr, 1.0, heatmap_color, alpha, 0)
    return superimposed_img

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or class_labels is None:
        return jsonify({'error': 'El modelo o los índices de clase no están cargados. Revisa los logs del servidor.'}), 500

    try:
        if 'file' in request.files:
            image = Image.open(request.files['file'].stream)
        elif 'url' in request.form:
            resp = requests.get(request.form['url'], timeout=10)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content))
        else:
            return jsonify({'error': 'Debes proporcionar una imagen a través de "file" o "url".'}), 400
    except Exception as e:
        return jsonify({'error': f'Error al leer la imagen: {e}'}), 400

    try:
        # Preprocesamiento y Predicción
        img_array = preprocess_image(image)
        pred_prob = float(model.predict(img_array)[0, 0])

        # Asignar etiqueta y confianza usando los índices cargados
        is_fractured = pred_prob < 0.5 # 'fractured' es la clase 0
        pred_label = class_labels[0] if is_fractured else class_labels[1]
        confidence = (1 - pred_prob) if is_fractured else pred_prob

        result = {
            'prediction': pred_label,
            'confidence_percent': round(confidence * 100, 2)
        }

        # Generar Grad-CAM solo si hay fractura
        if is_fractured:
            last_conv_layer_name = find_last_conv_layer(model)
            heatmap = get_gradcam_heatmap(img_array, last_conv_layer_name)
            
            orig_bgr = cv2.cvtColor(np.array(image.convert('RGB').resize((224, 224))), cv2.COLOR_RGB2BGR)
            annotated_img = superimpose_gradcam(orig_bgr, heatmap)

            _, buf = cv2.imencode('.jpg', annotated_img)
            img_b64 = base64.b64encode(buf).decode('utf-8')
            result['annotated_image'] = f"data:image/jpeg;base64,{img_b64}"

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error inesperado durante la predicción: {str(e)}'}), 500

if __name__ == "__main__":
    # Para producción, usa un servidor WSGI como Gunicorn:
    # gunicorn --workers 4 --bind 0.0.0.0:5000 api:app
    app.run(host='0.0.0.0', port=5000, debug=True)