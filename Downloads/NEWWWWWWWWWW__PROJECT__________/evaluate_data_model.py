import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


PROJECT_DIR = Path(__file__).parent
TEST_DIR = PROJECT_DIR / "Data" / "test"
MODEL_PATH = PROJECT_DIR / "data_classifier.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "data_class_names.json"
IMG_SIZE = (224, 224)


def canonical_name(name: str) -> str:
    s = name.lower()
    if "adenocarcinoma" in s:
        return "adenocarcinoma"
    if "large.cell.carcinoma" in s:
        return "large.cell.carcinoma"
    if "squamous.cell.carcinoma" in s:
        return "squamous.cell.carcinoma"
    if "normal" in s:
        return "normal"
    return s


def load_test_images():
    image_paths = []
    labels = []
    for class_dir in sorted(TEST_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        y = canonical_name(class_dir.name)
        for p in class_dir.glob("*"):
            if p.is_file():
                image_paths.append(p)
                labels.append(y)
    return image_paths, labels


def preprocess(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return arr


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run train_data_model.py first.")
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError("Class names file not found. Run train_data_model.py first.")

    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as f:
        model_class_names = json.load(f)

    class_to_idx = {c: i for i, c in enumerate(model_class_names)}
    canonical_order = sorted({canonical_name(c) for c in model_class_names})

    paths, y_true_names = load_test_images()
    if not paths:
        raise RuntimeError("No test images found in Data/test.")

    model = tf.keras.models.load_model(MODEL_PATH)
    x = np.stack([preprocess(p) for p in paths], axis=0)
    probs = model.predict(x, batch_size=32, verbose=1)
    pred_idx = np.argmax(probs, axis=1)
    pred_names = [canonical_name(model_class_names[i]) for i in pred_idx]

    y_true = [canonical_order.index(name) for name in y_true_names]
    y_pred = [canonical_order.index(name) for name in pred_names]

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=canonical_order, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    missing = [n for n in canonical_order if n not in {canonical_name(k) for k in class_to_idx}]
    if missing:
        print("Warning: Some test classes were not present in training classes:", missing)


if __name__ == "__main__":
    main()
