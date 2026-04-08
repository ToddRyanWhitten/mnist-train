# MNIST Training (venv)

This project trains a high-accuracy CNN on MNIST and saves the best model to `model/mnist_model.h5`.

## 1. Create and activate a virtual environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If activation worked, your shell prompt should show `(.venv)`.

## 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. (Optional) Configure Google Cloud credentials

If you want the trained model uploaded to GCS, you need a service account key file. If this is not set, training still runs and the model remains local.

### Create a service account key in GCP

1. Go to the [Google Cloud Console](https://console.cloud.google.com/) and select your project.
2. Navigate to **IAM & Admin > Service Accounts**.
3. Click **Create Service Account**, give it a name, and click **Done**.
4. Click the service account you just created, go to the **Keys** tab, and click **Add Key > Create new key**.
5. Choose **JSON** and click **Create** — the key file downloads automatically.
6. Grant the service account the **Storage Object Admin** role (or a more restrictive role scoped to your bucket) via **IAM & Admin > IAM**.

### Set the environment variable

Point `GOOGLE_CLOUD_CREDENTIALS` at the downloaded key file:

```bash
export GOOGLE_CLOUD_CREDENTIALS="/absolute/path/to/service-account.json"
```

## 4. Train

```bash
python train.py
```

## 5. Run the live demo

https://toddryanwhitten.github.io/mnist-train/

OR

Start a local HTTP server from the project root:

```bash
python -m http.server 8080
```

Then open [index.html](http://localhost:8080/index.html) in your browser to interact with the trained model via the live demo page.

## 6. Deactivate when done

```bash
deactivate
```

## Notes

- Model artifact path: `model/mnist_model.h5`
- Existing `model/` directory is reused if present.
- For VS Code, select the interpreter from `.venv/bin/python`.
