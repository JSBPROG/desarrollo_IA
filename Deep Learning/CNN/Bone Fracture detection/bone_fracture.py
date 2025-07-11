import os
import zipfile
import kagglehub
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#-------------------------------------------FUNCTIONS---------------------------

def download_and_unzip_dataset():
    """
    Download a dataset from Kaggle using kagglehub, unzip any .zip files found,
    and return the path where the dataset was extracted.
    """
    dataset_path = kagglehub.dataset_download("vuppalaadithyasairam/bone-fracture-detection-using-xrays")

    for item in os.listdir(dataset_path):
        if item.endswith(".zip"):
            zip_path = os.path.join(dataset_path, item)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(dataset_path)

    return dataset_path


def count_files_in_folder(folder_path):
    """
    Count total number of files inside folder and subfolders.
    """
    total_files = 0
    for root, dirs, files in os.walk(folder_path):
        total_files += len(files)
    return total_files


def show_random_images(dir_path, num_images=4):
    """
    Display a specified number of random images from a directory (recursively).
    """
    image_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))

    if len(image_files) < num_images:
        raise ValueError(f"Not enough images in folder to select {num_images}.")

    selected_files = random.sample(image_files, num_images)

    for file in selected_files:
        img = Image.open(file)
        plt.imshow(img)
        plt.axis('off')
        plt.show()


def load_image(path):
    """
    Load an image from disk using PIL.
    """
    return Image.open(path)


def resize_image(image, target_size=(224, 224)):
    """
    Resize a PIL image to the target size.
    """
    return image.resize(target_size)


def normalize_image(image_array):
    """
    Normalize a numpy image array to [0, 1].
    """
    return image_array.astype('float32') / 255.0


def pil_to_array(image):
    """
    Convert a PIL image to a numpy array.
    """
    return np.array(image)


def augment_images(image_array):
    """
    Apply random augmentations to an image array using Keras ImageDataGenerator.
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    image_array = np.expand_dims(image_array, 0)  # Add batch dimension
    aug_iter = datagen.flow(image_array, batch_size=1)
    augmented_image = next(aug_iter)[0].astype('float32')
    return augmented_image


def preprocess_for_model(image_path, target_size=(224, 224)):
    """
    Full pipeline: load, resize, convert to array, normalize for model input.
    """
    img = load_image(image_path)
    img = resize_image(img, target_size)
    img_array = pil_to_array(img)
    img_array = normalize_image(img_array)
    return img_array


#----------------------------------------CODE-----------------------------------

base_path = download_and_unzip_dataset() 
archive_folder = "archive (6)"

path_train_fractured = os.path.join(base_path, archive_folder, "train", "fractured")
path_train_not_fractured = os.path.join(base_path, archive_folder, "train", "not fractured")
path_val_fractured = os.path.join(base_path, archive_folder, "val", "fractured")
path_val_not_fractured = os.path.join(base_path, archive_folder, "val", "not fractured")

print(path_train_fractured)
print(path_train_not_fractured)
print(path_val_fractured)
print(path_val_not_fractured)

total_train_frac = count_files_in_folder(path_train_fractured)
total_train_nFrac = count_files_in_folder(path_train_not_fractured)
total_val_frac = count_files_in_folder(path_val_fractured)
total_val_nFrac = count_files_in_folder(path_val_not_fractured)

# Data visualization
counts = [
    total_train_frac,
    total_train_nFrac,
    total_val_frac,
    total_val_nFrac
]

labels = ['train_fractured', 'train_notFractured', 'val_fractured', 'val_notFractured']

plt.figure(figsize=(8, 5))
plt.barh(labels, counts, color='skyblue')
plt.xlabel('Number of Files')
plt.title('Image Counts per Dataset Split and Label')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Show example images
print("train_fractured examples:")
show_random_images(path_train_fractured)

print("train_not_fractured examples:")
show_random_images(path_train_not_fractured)

print("val_fractured examples:")
show_random_images(path_val_fractured)

print("val_not_fractured examples:")
show_random_images(path_val_not_fractured)

# Generating dataframes
df_fractured_train = pd.DataFrame({
    "file_path": [os.path.join(path_train_fractured, f) for f in os.listdir(path_train_fractured)],
    "label": "fractured"
})

df_notfractured_train = pd.DataFrame({
    "file_path": [os.path.join(path_train_not_fractured, f) for f in os.listdir(path_train_not_fractured)],
    "label": "not_fractured"
})

df_fractured_val = pd.DataFrame({
    "file_path": [os.path.join(path_val_fractured, f) for f in os.listdir(path_val_fractured)],
    "label": "fractured"
})

df_notfractured_val = pd.DataFrame({
    "file_path": [os.path.join(path_val_not_fractured, f) for f in os.listdir(path_val_not_fractured)],
    "label": "not_fractured"
})

print("Train - Fractured:")
print(df_fractured_train.head(), end="\n\n")

print("Train - Not Fractured:")
print(df_notfractured_train.head(), end="\n\n")

print("Validation - Fractured:")
print(df_fractured_val.head(), end="\n\n")

print("Validation - Not Fractured:")
print(df_notfractured_val.head())

df_train1 = pd.concat([df_fractured_train, df_notfractured_train], ignore_index=True)
df_val1 = pd.concat([df_fractured_val, df_notfractured_val], ignore_index=True)

# Dividing dataset

train_df, temp_df = train_test_split(
    df_train1,
    train_size=0.8,  
    shuffle=True,
    random_state=42,
)

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,  
    shuffle=True,
    random_state=42,
)

# Preprocessing parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# ImageDataGenerator for training with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

# For validation and test: only rescale
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='file_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=SEED
)

valid_generator = val_test_datagen.flow_from_dataframe(
    valid_df,
    x_col='file_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    test_df,
    x_col='file_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

#-------------------------CNN MODEL-----------------------------------

def build_cnn_model(input_shape=(224, 224, 3)):
    model = Sequential()

    # Conv Block 1
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Conv Block 2
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Conv Block 3
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))  # Binary classification output

    return model

model = build_cnn_model(input_shape=(224, 224, 3))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_bone_fracture_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=valid_generator,
    callbacks=[early_stopping, checkpoint]
)

# Evaluate on test set
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test precision: {test_precision:.4f}")
print(f"Test recall: {test_recall:.4f}")
