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

If you want the trained model uploaded to GCS, set:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/service-account.json"
```

If this is not set correctly, training still runs and the model remains local.

## 4. Train

```bash
python train.py
```

## 5. Deactivate when done

```bash
deactivate
```

## Notes

- Model artifact path: `model/mnist_model.h5`
- Existing `model/` directory is reused if present.
- For VS Code, select the interpreter from `.venv/bin/python`.
