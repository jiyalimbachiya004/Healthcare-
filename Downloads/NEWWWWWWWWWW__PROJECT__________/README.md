# Brain Tumor MRI Classification

This project trains a model on the `Training` dataset, evaluates it on the `Testing` dataset, and provides a Streamlit frontend for predictions.

## Dataset Structure

Keep folders exactly like this:

```text
Training/
  glioma/
  meningioma/
  notumor/
  pituitary/
Testing/
  glioma/
  meningioma/
  notumor/
  pituitary/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train_model.py
```

This creates:
- `brain_tumor_classifier.keras`
- `class_names.json`

## Test / Evaluate

```bash
python evaluate_model.py
```

This prints:
- Classification report (precision, recall, f1-score)
- Confusion matrix

## Run Frontend

```bash
streamlit run app.py
```

Open the local URL shown in terminal and upload an MRI image.

## Data Folder Pipeline

If you want to train and test using the `Data/` folder (`train`, `valid`, `test`):

```bash
python train_data_model.py
python evaluate_data_model.py
```

In frontend (`streamlit run app.py`), select:
- `Data Folder (train/valid/test)`
