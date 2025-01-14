# MNIST Digit Classification using Neural Networks

This repository contains a deep learning model built with *TensorFlow* and *Keras* to classify handwritten digits from the *MNIST* dataset. The model is trained using a simple neural network architecture, evaluated with accuracy metrics, and visualized using confusion matrices and other graphical representations.

## Overview

The *MNIST* dataset is a collection of 70,000 images of handwritten digits (0–9) used for training and testing image classification models. This project demonstrates the following:

- Loading and preprocessing the MNIST dataset.
- Building a feedforward neural network using *TensorFlow* and *Keras*.
- Evaluating the model's performance with various metrics.
- Visualizing the results with *Matplotlib* and *Seaborn*.

## Features

- *Model Architecture*: A simple neural network with fully connected layers.
- *Training*: 5 epochs of training using the Adam optimizer and categorical crossentropy loss.
- *Evaluation*: Test accuracy and per-class accuracy calculations.
- *Visualization*: 
    - Confusion matrix heatmap to visualize true vs. predicted labels.
    - Accuracy per digit class bar chart.
    - Displaying sample test images with predicted vs. true labels.


## Installation

### Prerequisites

To run this project, you need Python (3.6 or higher) installed. The following Python libraries are required:

- *TensorFlow*: For building and training the neural network model.
- *NumPy*: For numerical operations and array manipulation.
- *Matplotlib*: For plotting and visualizing data.
- *Seaborn*: For enhanced visualization, particularly confusion matrices.
- *scikit-learn*: For computing the confusion matrix and additional metrics.

### Install Dependencies

You can install all dependencies using the following command:

You can install them using pip:

```bash
pip install tensorflow
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
