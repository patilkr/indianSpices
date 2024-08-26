import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# Set parameters
batch_size = 32
img_height = 224
img_width = 224
dataset_dir = '/content/drive/MyDrive/Dataset/indian spices'  # Update this with your dataset path

# Function to get class names from directory
def get_class_names(directory):
    return sorted([d.name for d in os.scandir(directory) if d.is_dir()])

# Load and preprocess the dataset
val_ds_raw = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Capture class names from the directory
class_names = get_class_names(dataset_dir)

# Prepare the dataset for evaluation
AUTOTUNE = tf.data.AUTOTUNE
val_ds = val_ds_raw.cache().prefetch(buffer_size=AUTOTUNE)

def create_model(base_model):
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        Flatten(),
        Dense(len(class_names), activation='softmax')  # Dynamically set the number of classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize the models
models = {
    'VGG16': create_model(VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))),
    'ResNet50': create_model(ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))),
    'InceptionV3': create_model(InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3)))
}

def evaluate_model(model_name, model, val_ds, class_names):
    print(f"Evaluating {model_name}...")

    # Predict on validation data
    y_true = np.concatenate([y for x, y in val_ds], axis=0).flatten()
    y_pred = np.argmax(model.predict(val_ds), axis=1)

    # Generate classification report and confusion matrix
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Classification report for {model_name}:\n{report}\n")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

# Evaluate all models
for model_name, model in models.items():
    evaluate_model(model_name, model, val_ds, class_names)