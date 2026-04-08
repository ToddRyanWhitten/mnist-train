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

### Pass credentials at runtime (no key file in container)

You can pass service-account JSON directly in environment variables so the script never needs a credentials file path.

Option A (recommended): base64-encoded JSON

```bash
export GOOGLE_APPLICATION_CREDENTIALS_JSON_B64="$(base64 -w 0 ~/path/on/laptop/service-account.json)"
python train.py
unset GOOGLE_APPLICATION_CREDENTIALS_JSON_B64
```

Option B: raw JSON string

```bash
export GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat ~/path/on/laptop/service-account.json)"
python train.py
unset GOOGLE_APPLICATION_CREDENTIALS_JSON
```

The script checks auth in this order:
1. `GOOGLE_APPLICATION_CREDENTIALS_JSON_B64`
2. `GOOGLE_APPLICATION_CREDENTIALS_JSON`
3. `GOOGLE_APPLICATION_CREDENTIALS` (file path)

## 4. Train

```bash
python train.py
```

## 5. Run the live demo

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
