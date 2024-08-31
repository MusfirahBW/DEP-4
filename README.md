# DEP-4
Machine Learning Simple Model for Image Classification

## Objective:
Build a model to classify images into different categories using Convolutional Neural Networks (CNN).

## Description:
Use a labeled dataset of images, such as CIFAR-10, to train a deep learning model that can classify images into predefined categories.

## Key Steps:

1. Data Augmentation and Preprocessing:
   - Load the CIFAR-10 dataset.
   - Normalize pixel values to scale them between 0 and 1.
   - Apply data augmentation techniques such as rotation, width shift, height shift, and horizontal flip.

2. Building the Convolutional Neural Network (CNN):
   - Design a CNN architecture with convolutional layers followed by pooling layers.
   - Use a dropout layer to prevent overfitting.
   - Implement a final softmax layer for multi-class classification.

3. Model Training:
   - Compile the model using the Adam optimizer and categorical cross-entropy loss function.
   - Train the model using the training dataset with data augmentation.
   - Validate the model using the test dataset.

4. Model Evaluation:
   - Evaluate model performance using accuracy on the test dataset.
   - Visualize training and validation accuracy over epochs.

5. Saving the Model:
   - Save the trained model for future use.

## Language:
Python

## Dependencies:

- TensorFlow
- Keras (included in TensorFlow)
- Matplotlib
- NumPy
 
