import kagglehub
import zipfile
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
import numpy as np

# Variables globales por defecto (vacías)
route_test = None
route_train = None
route_val = None

#--------------------------------FUNCTIONS--------------------

def download_data_and_unzip():
    """
    Descarga y descomprime el dataset desde Kaggle Hub.
    Devuelve el path base donde quedan los datos.
    """
    dataset_identifier = "ashfakyeafi/glasses-classification-dataset"
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
        extract_dir = os.path.join(os.path.dirname(zip_file_found), "extracted_dataset")
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


def get_dataset_paths():
    """
    Descarga (si es necesario) y devuelve las rutas de test, train y validate.
    Returns:
        tuple: (route_test, route_train, route_val)
    """
    path_base = download_data_and_unzip()
    route_test = os.path.join(path_base, "test")
    route_train = os.path.join(path_base, "train")
    route_val = os.path.join(path_base, "validate")
    return route_test, route_train, route_val


def predict_glasses(img_name, model, route_test):
    """
    Predice si la imagen tiene gafas usando el modelo.
    """
    img_path = os.path.join(route_test, img_name)
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return "SIN GAFAS" if prediction[0][0] >= 0.5 else "GAFAS"


#----------------------------------PROTECTED CODE----------------------
if __name__ == "__main__":
    # Descargar datos y configurar rutas
    path_base = download_data_and_unzip()
    route_test = os.path.join(path_base, "test")
    route_train = os.path.join(path_base, "train")
    route_val = os.path.join(path_base, "validate")

    # Crear modelo
    classifier = Sequential()
    classifier.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(64,64,3), activation="relu"))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation="relu"))
    classifier.add(Dense(units=64, activation="relu"))
    classifier.add(Dense(units=32, activation="relu"))
    classifier.add(Dense(units=1, activation="sigmoid"))

    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Generadores de datos
    batch_size = 5

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        route_train,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        route_val,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary')

    # Entrenar modelo
    classifier.fit(
        train_generator,
        steps_per_epoch=100 // batch_size,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

    # Guardar modelo
    classifier.save_weights('first_try.weights.h5')
    classifier.save('first_try.h5')
