# Multi-Layer Neural Network with TensorFlow

Author: Nikhil Das Karavatt
Date: 2023-03-19


## Description

This repository contains Python code for building and training a multi-layer neural network using TensorFlow. The network architecture includes options for specifying the number of layers, activation functions, learning rate, batch size, and training epochs. It provides support for Mean Squared Error (MSE), Support Vector Machine (SVM), and Cross-Entropy loss functions.

## Functions

- `activation_function(activations)`: Choose an activation function for each layer (sigmoid, linear, or relu).

- `split_data(X_train, Y_train, split_range)`: Helper function to split data into training and validation sets.

- Activation functions: Sigmoid, Linear, ReLU.

- `add_bias_layer(X)`: Add bias to input matrices.

- `initial_weights(X_train, layers, seed)`: Initialize network weights.

- `Activation(weights, X_batch, layers, activations)`: Compute activated values for each layer.

- `output(weights, X_batch, layers, activations)`: Get the network's output.

- `cost(loss, Y_pred, Y_batch)`: Calculate the loss (MSE, SVM, or Cross-Entropy).

- `training(weights, X_batch, Y_batch, layers, activations, alpha, loss)`: Train the neural network using backpropagation.

- `multi_layer_nn_tensorflow(X_train, Y_train, layers, activations, alpha, batch_size, epochs, loss, validation_split, weights, seed)`: Train a multi-layer neural network using TensorFlow.

## Usage

You can customize the neural network's architecture, learning parameters, and loss function by modifying the provided functions. Use the `multi_layer_nn_tensorflow` function to train and evaluate your model.

Example usage:
```python
# Define network architecture and parameters
layers = [64, 32, 16, 1]
activations = ["relu", "relu", "sigmoid"]
alpha = 0.01
batch_size = 32
epochs = 100
loss = "mse"
validation_split = [0.8, 1.0]

# Train the model
weights, mse_per_epoch, predictions = multi_layer_nn_tensorflow(
    X_train, Y_train, layers, activations, alpha, batch_size, epochs, loss, validation_split
)

# Evaluate the model

