from glasses_classification import predict_glasses, get_dataset_paths
from tensorflow.keras.models import load_model

# Llamar a get_dataset_paths() para inicializar las rutas
route_test, _, _ = get_dataset_paths()

# Cargar el modelo entrenado
model = load_model('first_try.h5')

# Hacer predicciones
print(predict_glasses("fotogafa.jpg", model, route_test=route_test))  #Si
print(predict_glasses("row-7-column-6.jpg", model, route_test=route_test))  #sin gafas
print(predict_glasses("fotogafa2.jpg", model, route_test=route_test)) #si
