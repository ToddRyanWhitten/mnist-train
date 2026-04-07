# MNIST Training with TensorFlow

## Overview
This project trains a neural network on the MNIST dataset using TensorFlow to classify handwritten digits (0-9).

## Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib (for visualization)

## Installation
```bash
pip install tensorflow numpy matplotlib
```

## Dataset
The MNIST dataset contains 70,000 images of handwritten digits:
- 60,000 training images
- 10,000 test images
- Image size: 28×28 pixels
- Labels: 0-9

## Training
```python
import tensorflow as tf

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate
model.evaluate(x_test, y_test)
```

## Results
Expected accuracy: ~97-98% on test set

## References
- [TensorFlow MNIST Guide](https://www.tensorflow.org/datasets/catalog/mnist)
- [Keras API Documentation](https://keras.io/)