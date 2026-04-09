import os
from google.cloud import storage
from google.oauth2 import service_account

CREDENTIALS_FILE = "keys/mnist.json"
DESTINATION_BUCKET_NAME = "whitten-data-bucket"

def upload_model():

    SOURCE_FILE = "models/mnist/v0.0.1/mnist.h5"
    DESTINATION_FILEPATH = "models/mnist/v0.0.1/mnist.h5"

    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(DESTINATION_BUCKET_NAME)
    blob = bucket.blob(DESTINATION_FILEPATH)

    blob.upload_from_filename(SOURCE_FILE)
    print(f"Uploaded {SOURCE_FILE} to gs://{DESTINATION_BUCKET_NAME}/{DESTINATION_FILEPATH}")

def upload_js():

    SOURCE_DIR = "models/mnist"
    DESTINATION_PREFIX = "models/mnist"

    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(DESTINATION_BUCKET_NAME)

    for root, _, files in os.walk(SOURCE_DIR):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, SOURCE_DIR)
            destination_path = f"{DESTINATION_PREFIX}/{relative_path}"
            blob = bucket.blob(destination_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to gs://{DESTINATION_BUCKET_NAME}/{destination_path}")

if __name__ == "__main__":
    upload_model()
    upload_js()
