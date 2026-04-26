#!/usr/bin/env python3
"""
MNIST Training Script with Optimized CNN Architecture - TPU Version
Runs on Google Cloud TPUs via tf.distribute.TPUStrategy.

Usage:
    python train_tpu.py --gcs-bucket gs://your-bucket --tpu-name your-tpu

The GCS bucket is required because TPUs cannot write to local disk.
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras  # type: ignore

layers = keras.layers

np.random.seed(42)
tf.random.set_seed(42)


def connect_to_tpu(tpu_name: str):
    """Resolves the TPU and initializes the TPUStrategy."""
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print(f"Connected to TPU: {tpu_name}")
    print(f"Number of replicas: {strategy.num_replicas_in_sync}")
    return strategy


def create_model(strategy):
    """Creates the same CNN model as train.py, built within the TPU strategy scope."""
    with strategy.scope():
        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),

            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(10, activation='softmax'),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

    return model


def load_and_preprocess_data():
    """Loads and preprocesses MNIST; identical logic to train.py."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test  = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    print(f"Training samples : {x_train.shape[0]}")
    print(f"Test samples     : {x_test.shape[0]}")
    print(f"Image shape      : {x_train.shape[1:]}")

    return (x_train, y_train), (x_test, y_test)


def train_model(model, x_train, y_train, x_test, y_test, gcs_bucket, num_replicas):
    """Trains the model with the same callbacks as train.py; checkpoints go to GCS."""
    # TPUs cannot write to local disk -- model artifacts must live in GCS.
    model_dir  = os.path.join(gcs_bucket, "mnist_tpu")
    model_path = os.path.join(model_dir, "mnist_model.keras")

    # Scale batch size linearly with the number of TPU cores.
    batch_size = 128 * num_replicas

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    print(f"\nStarting training  (batch_size={batch_size})...")
    print("=" * 60)

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=30,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    return history, model_path


def evaluate_model(model, x_test, y_test):
    """Evaluates the trained model; identical logic to train.py."""
    print("\n" + "=" * 60)
    print("Evaluating model on test data...")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

    y_pred         = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\nPer-class accuracy:")
    for i in range(10):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_classes[mask] == i)
            print(f"  Digit {i}: {class_acc * 100:.2f}%")

    return test_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Train MNIST on Cloud TPU")
    parser.add_argument(
        "--tpu-name",
        default=os.environ.get("TPU_NAME", "local"),
        help="TPU name or 'local' for TPU VM (default: $TPU_NAME or 'local')",
    )
    parser.add_argument(
        "--gcs-bucket",
        default=os.environ.get("GCS_BUCKET"),
        required=not os.environ.get("GCS_BUCKET"),
        help="GCS bucket URI for checkpoints, e.g. gs://my-bucket (or set $GCS_BUCKET)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("MNIST High-Accuracy Training Script  [TPU]")
    print("=" * 60)
    print(f"TPU name   : {args.tpu_name}")
    print(f"GCS bucket : {args.gcs_bucket}")

    strategy = connect_to_tpu(args.tpu_name)

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    print("\nCreating optimized CNN model...")
    model = create_model(strategy)
    model.summary()

    history, model_path = train_model(
        model, x_train, y_train, x_test, y_test,
        gcs_bucket=args.gcs_bucket,
        num_replicas=strategy.num_replicas_in_sync,
    )

    print(f"\nLoading best model from {model_path}...")
    with strategy.scope():
        best_model = keras.models.load_model(model_path)

    test_accuracy = evaluate_model(best_model, x_test, y_test)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to : {model_path}")
    print(f"Test accuracy  : {test_accuracy * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
