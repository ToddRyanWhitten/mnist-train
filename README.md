# MNIST Training

This project trains a high-accuracy CNN on MNIST and saves the best model to `model/mnist_model.h5`.

## Development container

The recommended way to develop and run this project is with the included Dev Container. When the container starts, `devcontainer.json` runs `setup.sh` via `postCreateCommand`, which automatically creates the `.venv` virtual environment and installs all dependencies from `requirements.txt`. No manual setup is needed.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- VS Code with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Getting started

1. Open the repository in VS Code.
2. When prompted, click **Reopen in Container** (or run **Dev Containers: Reopen in Container** from the command palette).
3. VS Code builds the image from `Dockerfile`, starts the container, and runs `setup.sh` — the environment is ready when the terminal prompt appears.

The container is built from `tensorflow/tensorflow:2.21.0` and includes `tensorflowjs`. Your GCS credentials file is bind-mounted read-only from `~/.config/mnist.json` on the host into `keys/mnist.json` inside the container.

## Train

```bash
python train.py
```

## (Optional) Google Cloud credentials

If you want the trained model uploaded to GCS, ensure your service account key is in place at `~/.config/mnist.json` on the host before starting the container (it is mounted automatically). If the key is absent, training still runs and the model remains local.

To create a service account key:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/) and select your project.
2. Navigate to **IAM & Admin > Service Accounts**.
3. Click **Create Service Account**, give it a name, and click **Done**.
4. Click the account, go to the **Keys** tab, and click **Add Key > Create new key**.
5. Choose **JSON** and click **Create** — the key file downloads automatically.
6. Grant the service account the **Storage Object Admin** role via **IAM & Admin > IAM**.
7. Save the key to `~/.config/mnist.json` on your host.

## Run the live demo

https://toddryanwhitten.github.io/mnist-train/

OR start a local HTTP server from the project root:

```bash
python -m http.server 8080
```

Then open [http://localhost:8080/index.html](http://localhost:8080/index.html) in your browser.

## Notes

- Model artifact path: `model/mnist_model.h5`
- Existing `model/` directory is reused if present.
- `setup.sh` is idempotent — it is safe to re-run manually if needed.
