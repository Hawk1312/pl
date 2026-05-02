import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "PlantVillage"
VERIFIED_DATASET_DIR = BASE_DIR / "verified_disease_dataset"
MODEL_PATH = BASE_DIR / "plant_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
METRICS_PATH = BASE_DIR / "model_metrics.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS_HEAD = 12
EPOCHS_FINE_TUNE = 10
VALIDATION_SPLIT = 0.20


def normalize_class_name(path: Path, dataset_root: Path) -> str:
    relative = path.relative_to(dataset_root)
    parts = [part for part in relative.parts if part.lower() != "tomato"]
    return "__".join(parts).replace(" ", "_")


def find_class_folders(dataset_dir: Path):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    class_folders = []

    for folder in sorted(dataset_dir.rglob("*")):
        if not folder.is_dir():
            continue

        has_images = any(
            file.is_file() and file.suffix.lower() in image_extensions
            for file in folder.iterdir()
        )
        if has_images:
            class_folders.append(folder)

    return class_folders


def build_file_dataset():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    dataset_roots = [DATASET_DIR]
    if VERIFIED_DATASET_DIR.exists():
        dataset_roots.append(VERIFIED_DATASET_DIR)

    class_folder_records = []
    for dataset_root in dataset_roots:
        for class_folder in find_class_folders(dataset_root):
            class_folder_records.append((dataset_root, class_folder))

    class_names = sorted(
        {
            normalize_class_name(class_folder, dataset_root)
            for dataset_root, class_folder in class_folder_records
        }
    )
    class_to_index = {name: index for index, name in enumerate(class_names)}

    image_paths = []
    labels = []
    for dataset_root, folder in class_folder_records:
        class_name = normalize_class_name(folder, dataset_root)
        label = class_to_index[class_name]
        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                image_paths.append(str(image_path))
                labels.append(label)

    with CLASS_NAMES_PATH.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    return image_paths, labels, class_names


def load_image(image_path, label):
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def make_datasets(image_paths, labels):
    path_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    path_ds = path_ds.shuffle(len(image_paths), seed=SEED, reshuffle_each_iteration=False)

    validation_size = int(len(image_paths) * VALIDATION_SPLIT)
    val_ds = path_ds.take(validation_size)
    train_ds = path_ds.skip(validation_size)

    train_ds = (
        train_ds
        .shuffle(2048, seed=SEED)
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds


def build_model(num_classes: int):
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.12),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
        ],
        name="leaf_augmentation",
    )

    base_model = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        include_preprocessing=False,
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0, name="leaf_rescaling")(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def train():
    image_paths, labels, class_names = build_file_dataset()
    train_ds, val_ds = make_datasets(image_paths, labels)
    model, base_model = build_model(len(class_names))

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            verbose=1,
        ),
    ]

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-35]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_fine_tune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE_TUNE,
        callbacks=callbacks,
    )

    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
    model.save(MODEL_PATH)

    metrics = {
        "image_size": list(IMG_SIZE),
        "num_classes": len(class_names),
        "validation_loss": float(val_loss),
        "validation_accuracy": float(val_accuracy),
        "head_epochs": len(history_head.history["loss"]),
        "fine_tune_epochs": len(history_fine_tune.history["loss"]),
        "architecture": "EfficientNetV2B0 transfer learning",
    }
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


if __name__ == "__main__":
    train()
