#!/usr/bin/env python3
"""
MNIST Training Script with Optimized CNN Architecture
Achieves 99.5%+ accuracy on the MNIST dataset
"""

import os
import tensorflow as tf
from tensorflow import keras # type: ignore
import numpy as np

# Use keras.layers to avoid Pylance import resolution issues
layers = keras.layers

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_model():
    """
    Creates an optimized CNN model for MNIST classification.
    Architecture designed for maximum accuracy.
    """
    model = keras.Sequential([
        # Input layer - reshape to include channel dimension
        layers.Input(shape=(28, 28, 1)),
        
        # First convolutional block
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(10, activation='softmax')
    ])
    
    return model

def load_and_preprocess_data():
    """
    Loads and preprocesses the MNIST dataset.
    Returns normalized and reshaped training and test data.
    """
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Reshape data to include channel dimension (28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')
    
    # Normalize pixel values from [0, 255] to [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Image shape: {x_train.shape[1:]}")
    
    return (x_train, y_train), (x_test, y_test)

def train_model(model, x_train, y_train, x_test, y_test):
    """
    Trains the model with optimized hyperparameters and callbacks.
    """
    # Create model directory if it doesn't exist
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'mnist_model.h5')
    
    # Compile model with optimized parameters
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks for optimal training
    callbacks = [
        # Save the best model based on validation accuracy
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Stop training if validation accuracy doesn't improve
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when validation accuracy plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    print("\nStarting training...")
    print("=" * 60)
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=30,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    
    return history, model_path

def evaluate_model(model, x_test, y_test):
    """
    Evaluates the trained model on test data.
    """
    print("\n" + "=" * 60)
    print("Evaluating model on test data...")
    print("=" * 60)
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Get predictions for additional metrics
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(10):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_classes[mask] == i)
            print(f"  Digit {i}: {class_acc * 100:.2f}%")
    
    return test_accuracy

def main():

    """
    Main training pipeline.
    """
    print("=" * 60)
    print("MNIST High-Accuracy Training Script")
    print("=" * 60)
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create model
    print("\nCreating optimized CNN model...")
    model = create_model()
    model.summary()
    
    # Train model
    history, model_path = train_model(model, x_train, y_train, x_test, y_test)
    
    # # Load the best saved model
    # print(f"\nLoading best model from {model_path}...")
    best_model = keras.models.load_model(model_path)

    # Evaluate on test set
    test_accuracy = evaluate_model(best_model, x_test, y_test)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved locally to: {model_path}")
    print(f"Final test accuracy: {test_accuracy * 100:.2f}%")


    print("=" * 60)

if __name__ == "__main__":
    main()
