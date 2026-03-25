import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


DATA_DIR = Path(__file__).parent
TEST_DIR = DATA_DIR / "Testing"
MODEL_PATH = DATA_DIR / "brain_tumor_classifier.keras"
CLASS_NAMES_PATH = DATA_DIR / "class_names.json"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run train_model.py first.")
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError("class_names.json not found. Run train_model.py first.")

    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as f:
        class_names = json.load(f)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = tf.keras.models.load_model(MODEL_PATH)

    y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
