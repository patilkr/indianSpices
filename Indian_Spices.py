import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import os

# Set parameters
batch_size = 32
img_height = 224
img_width = 224
epochs = 5
dataset_dir = '/content/drive/MyDrive/Dataset/indian spices'  # Update this with your dataset path
steps_per_epoch = 100  # Adjust the number of steps per epoch

# Function to get class names from directory
def get_class_names(directory):
    return sorted([d.name for d in os.scandir(directory) if d.is_dir()])

# Load and preprocess the dataset
train_ds_raw = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

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

# Prepare the dataset for training
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds_raw.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds_raw.cache().prefetch(buffer_size=AUTOTUNE)

# Limit the number of steps per epoch
train_ds = train_ds.take(steps_per_epoch)
val_ds = val_ds.take(steps_per_epoch)

def create_model(base_model):
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        Flatten(),
        Dense(len(class_names), activation='softmax')  # Dynamically set the number of classes
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

models = {
    'VGG16': create_model(VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))),
    'ResNet50': create_model(ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))),
    'InceptionV3': create_model(InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3)))
}

def train_and_evaluate(model_name, model, train_ds, val_ds, class_names):
    print(f"Training {model_name}...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save(f'/content/{model_name}.h5')
    print(f"{model_name} model saved.")

    # Plot accuracy
    plt.plot(history.history['accuracy'], label=f'{model_name} Train Accuracy')
    plt.plot(history.history['val_accuracy'], label=f'{model_name} Val Accuracy')

    # Classification report and confusion matrix
    y_true = np.concatenate([y for x, y in val_ds], axis=0).flatten()
    y_pred = np.argmax(model.predict(val_ds), axis=1)
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

# Train and evaluate all models
plt.figure(figsize=(12, 8))
for model_name, model in models.items():
    train_and_evaluate(model_name, model, train_ds, val_ds, class_names)
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()