import json
from pathlib import Path

import tensorflow as tf


DATA_DIR = Path(__file__).parent / "Data"
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
MODEL_PATH = Path(__file__).parent / "data_classifier.keras"
CLASS_NAMES_PATH = Path(__file__).parent / "data_class_names.json"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8
SEED = 42


def build_model(num_classes: int) -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    if not TRAIN_DIR.exists() or not VALID_DIR.exists():
        raise FileNotFoundError("Data/train or Data/valid folders were not found.")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
        VALID_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_names=train_ds.class_names,
    )

    class_names = train_ds.class_names
    with CLASS_NAMES_PATH.open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    valid_ds = valid_ds.prefetch(autotune)

    model = build_model(num_classes=len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(valid_ds, verbose=0)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Saved model at: {MODEL_PATH}")
    print(f"Saved class names at: {CLASS_NAMES_PATH}")


if __name__ == "__main__":
    tf.random.set_seed(SEED)
    main()
