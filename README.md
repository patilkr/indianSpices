# indianSpices
The goal of this project is to create a model that can classify images of Indian spices into their respective categories. The project utilizes several popular CNN architectures such as VGG16, ResNet50, and InceptionV3 for feature extraction and classification.

# Project Overview
The goal of this project is to create a model that can classify images of Indian spices into their respective categories. The project utilizes several popular CNN architectures such as VGG16, ResNet50, and InceptionV3 for feature extraction and classification.

# Requirements
To run this project, you need the following dependencies:
  `Python 3.x`,
  `TensorFlow`,
  `NumPy`,
  `Matplotlib`,
  `Seaborn`,
  `scikit-learn`

You can install the required packages using the following command:
     `pip install tensorflow numpy matplotlib seaborn scikit-learn`

# Dataset
  ## Repository name: Indian Spices Image Dataset
  ## Data identification number: `10.17632/vg77y9rtjb.1`
  ## Direct URL to data  : `https://data.mendeley.com/datasets/vg77y9rtjb/1`

# Model Architecture
The project leverages the following pre-trained models for feature extraction:
  `VGG16`,
  `ResNet50`,
  `InceptionV3`
These models are used as the base for transfer learning, where the final layers are fine-tuned to classify the spices.

To train the model, the following parameters are used (which are adjustable):
  **Batch Size: 32
  Image Height: 224
  Image Width: 224
  Epochs: 5**

# Evaluation
The performance of the model is evaluated using classification reports and confusion matrices. These metrics provide insights into the accuracy and precision of the model across different spice categories.

# Usage
To train and evaluate the model, execute the Python scripts provided:
  ## Before Training: 
  The script **Before_Tranning.py** includes the setup and preparation steps before the actual model training.
  ## Model Training and Evaluation: 
  The script **Indian_Spices.py** contains the code for training the model and evaluating its performance.

Make sure the dataset is correctly placed and the paths in the scripts are updated to match your environment.
