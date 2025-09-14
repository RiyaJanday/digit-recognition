#Handwritten Digit Recognition using CNN

#Objective:
The objective of this project was to develop a machine learning model that can accurately recognize handwritten digits using the MNIST dataset. The project also includes an interactive user interface built with Streamlit where users can draw digits and get real-time predictions using the trained model.

#Tools and Technologies Used:
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV
- Streamlit
- streamlit-drawable-canvas
- Pillow (PIL)

#Dataset:
- Name: MNIST Handwritten Digit Dataset
- Source: tensorflow.keras.datasets
- Images: 70,000 grayscale images (28x28 pixels)
- Training: 60,000
- Testing: 10,000

#Project Files:
- digit_recognition.py: Script to train a Convolutional Neural Network (CNN) and save the trained model.
- draw_ui.py: Streamlit-based user interface for drawing digits and receiving predictions.
- models/digit_cnn_model.h5: The saved trained model file.

#Model Architecture:
The CNN model consists of the following layers:
- Conv2D (32 filters, 3x3 kernel, ReLU)
- MaxPooling2D (2x2)
- Conv2D (64 filters, 3x3 kernel, ReLU)
- MaxPooling2D (2x2)
- Flatten
- Dense (64 units, ReLU)
- Output Dense (10 units, Softmax)

#Training Details:
- Epochs: 5
- Batch Size: 64
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Final Test Accuracy: Approximately 98%

#How to Run:

1. Train the Model:

To retrain the model:

python digit_recognition.py

2. Launch the Web Interface

#To run the Streamlit digit recognition app:

run draw_ui.py

#Requirements:

#Install all required libraries using pip:

tensorflow 
numpy 
matplotlib 
opencv-python 
streamlit 
streamlit-drawable-canvas 
Pillow

#Output Samples:

training\_plot.png**: Accuracy graph of training and validation data.
model\_accuracy.png**: Screenshot of test accuracy printed in terminal.
prediction\_interface.png**: Screenshot of the Streamlit app displaying predicted digit.
