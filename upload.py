from google.cloud import storage
from google.oauth2 import service_account

CREDENTIALS_FILE = "keys/mnist.json"
BUCKET_NAME = "whitten-data-bucket"
SOURCE_FILE = "model/mnist_model.h5"
DESTINATION_BLOB = "models/mnist/mnist_model.h5"


def upload_model():
    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(DESTINATION_BLOB)

    blob.upload_from_filename(SOURCE_FILE)
    print(f"Uploaded {SOURCE_FILE} to gs://{BUCKET_NAME}/{DESTINATION_BLOB}")


if __name__ == "__main__":
    upload_model()
