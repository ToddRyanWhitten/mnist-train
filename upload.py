from google.cloud import storage
from google.oauth2 import service_account

CREDENTIALS_FILE = "keys/mnist.json"
DESTINATION_BUCKET_NAME = "whitten-data-bucket"

def upload_model():

    SOURCE_FILE = "model/mnist_model.h5"
    DESTINATION_FILEPATH = "models/mnist/mnist_model.h5"

    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(DESTINATION_BUCKET_NAME)
    blob = bucket.blob(DESTINATION_FILEPATH)

    blob.upload_from_filename(SOURCE_FILE)
    print(f"Uploaded {SOURCE_FILE} to gs://{DESTINATION_BUCKET_NAME}/{DESTINATION_FILEPATH}")

# def upload_js():



if __name__ == "__main__":
    upload_model()
