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
pip install tensorflow numpy matplotlib google-cloud-storage
```

## Dataset
The MNIST dataset contains 70,000 images of handwritten digits:
- 60,000 training images
- 10,000 test images
- Image size: 28×28 pixels
- Labels: 0-9

## Training


## Google Cloud Storage Integration

The training script automatically uploads trained models to Google Cloud Storage with the accuracy included in the filename.

gs://whitten-data-bucket/models/mnist/

gs://whitten-data-bucket/models/mnist/mnist_model_99.48.h5

### Setup

1. **Create a Service Account** in Google Cloud Console:
   - Go to IAM & Admin > Service Accounts
   - Create a new service account with "Storage Admin" role
   - Download the JSON key file

2. **Set Environment Variable**:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account-key.json"
   ```

3. **Run Training**:
   ```bash
   python train.py
   ```

### Upload Details

- **Bucket**: `whitten-data-bucket`
- **Path**: `models/mnist/`
- **Filename Format**: `mnist_model_{accuracy}.h5` (e.g., `mnist_model_99.21.h5`)
- **Console URL**: https://console.cloud.google.com/storage/browser/whitten-data-bucket/models/mnist

### Error Handling

If the upload fails (e.g., missing credentials or network issues):
- The error will be logged to the console
- Training will continue without interruption
- The model will still be saved locally in the `model/` directory

## Results
Expected accuracy: ~97-98% on test set (basic model)
Expected accuracy: ~99%+ on test set (optimized CNN in train.py)

## References
- [TensorFlow MNIST Guide](https://www.tensorflow.org/datasets/catalog/mnist)
- [Keras API Documentation](https://keras.io/)