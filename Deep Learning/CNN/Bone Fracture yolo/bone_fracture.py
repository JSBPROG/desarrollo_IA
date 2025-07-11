import kagglehub
import os
import zipfile
from ultralytics import YOLO

#----------------------------------
# Descargar y descomprimir dataset
#----------------------------------
def download_data_and_unzip():
    dataset_identifier = "pkdarabi/bone-fracture-detection-computer-vision-project"
    path_to_downloaded_data = kagglehub.dataset_download(dataset_identifier)
    print("Path to downloaded dataset files:", path_to_downloaded_data)

    zip_file_found = None
    for root, _, files in os.walk(path_to_downloaded_data):
        for file in files:
            if file.endswith(".zip"):
                zip_file_found = os.path.join(root, file)
                break
        if zip_file_found:
            break

    if zip_file_found:
        print(f"Found zip file: {zip_file_found}")
        extract_dir = os.path.join(os.path.dirname(zip_file_found), "BoneFractureYolo8")
        os.makedirs(extract_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_file_found, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Successfully unzipped to: {extract_dir}")
            return extract_dir
        except zipfile.BadZipFile:
            print(f"Error: {zip_file_found} is not a valid zip file or is corrupted.")
            return None
        except Exception as e:
            print(f"An error occurred during unzipping: {e}")
            return None
    else:
        print("No zip file found. Dataset might have been downloaded as uncompressed files directly.")
        return path_to_downloaded_data


#----------------------------------
# Entrenar modelo YOLOv8
#----------------------------------
if __name__ == "__main__":
    # Descargar y preparar dataset
    dataset_path = download_data_and_unzip()
    if not dataset_path:
        print("Dataset no disponible.")
        exit(1)

    # Ruta al data.yaml
    data_yaml = os.path.join(dataset_path, "data.yaml")

    # Crear el archivo data.yaml si no existe
    if not os.path.exists(data_yaml):
        with open(data_yaml, "w") as f:
            f.write(f"""
train: {os.path.join(dataset_path, 'BoneFractureYolo8/train/images')}
val: {os.path.join(dataset_path, 'BoneFractureYolo8/valid/images')}
test: {os.path.join(dataset_path, 'BoneFractureYolo8/test/images')}

nc: 1
names: ['fracture']
""")
        print("Archivo data.yaml creado.")

    # Crear y entrenar modelo
    model = YOLO("yolov8n.pt")  

    model.train(data=data_yaml, epochs=25, imgsz=640)

    # Evaluar modelo
    metrics = model.val()
    print("Resultados de validación:", metrics)

    # Prueba rápida sobre imágenes de test
    test_images_dir = os.path.join(dataset_path, "BoneFractureYolo8/test/images")
    test_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) if f.endswith(".jpg")]

    if test_images:
        results = model(test_images[:5], save=True)  # Guarda predicciones como imágenes
        print("Predicciones hechas y guardadas en runs/detect/predict")
    else:
        print("No se encontraron imágenes de test para probar.")

    # Guardar modelo final (opcional)
    model.export(format="onnx")
    print("Entrenamiento y exportación completados.")
