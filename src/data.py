
import os
import re
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight


CLASSES = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}


def load_records(dataset_path):
    """
    Load image paths and labels from train, val, and test folders.
    Expected structure:
        dataset_path/
            train/
                NORMAL/
                PNEUMONIA/
            val/
                NORMAL/
                PNEUMONIA/
            test/
                NORMAL/
                PNEUMONIA/
    """

    records = []

    split_names = {
        "train": "train",
        "val": "val",
        "validation": "val",
        "test": "test"
    }

    for folder_name, split_name in split_names.items():

        split_path = os.path.join(dataset_path, folder_name)

        if not os.path.isdir(split_path):
            continue

        for class_name, label in CLASSES.items():

            class_path = os.path.join(split_path, class_name)

            if not os.path.isdir(class_path):
                continue

            for filename in os.listdir(class_path):

                file_path = os.path.join(class_path, filename)

                if os.path.isfile(file_path):
                    records.append({
                        "path": file_path,
                        "label": label,
                        "class_name": class_name,
                        "split": split_name,
                        "filename": filename
                    })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            f"No images found in dataset directory: {dataset_path}"
        )

    return df


def clean_data(df):
    """
    Remove unreadable images and exact duplicate files.
    Test split has priority when duplicates exist.
    """

    valid_records = []

    for _, row in df.iterrows():

        try:
            with Image.open(row["path"]) as img:
                img.verify()

            with open(row["path"], "rb") as file:
                file_hash = hashlib.sha256(file.read()).hexdigest()

            record = row.to_dict()
            record["hash"] = file_hash

            valid_records.append(record)

        except Exception:
            print(f"Bad image removed: {row['path']}")

    clean = pd.DataFrame(valid_records)

    if clean.empty:
        raise ValueError("No valid images remain after cleaning.")

    # Check that identical files do not have different labels.
    label_counts = clean.groupby("hash")["label"].nunique()

    conflicting_hashes = label_counts[label_counts > 1]

    if not conflicting_hashes.empty:
        raise ValueError(
            "Identical image files were assigned different labels."
        )

    # Test has priority, followed by validation, then train.
    priority = {
        "test": 0,
        "val": 1,
        "train": 2
    }

    clean["priority"] = clean["split"].map(priority)

    clean = clean.sort_values("priority")

    clean = clean.drop_duplicates(
        subset="hash",
        keep="first"
    )

    clean = clean.drop(columns=["priority"])

    return clean.reset_index(drop=True)


def extract_patient_id(filename):
    """
    Extract patient ID from common chest X-ray filename patterns.
    """

    match = re.search(
        r"(person\d+|(?:NORMAL2-)?IM-\d+)",
        filename,
        re.IGNORECASE
    )

    if match:
        return match.group(1).lower()

    # Fallback: use filename without extension.
    return os.path.splitext(filename)[0]


def split_data(df, seed=42):
    """
    Keep the original test split.
    Split train + validation using patient-level grouping.
    """

    test_df = df[df["split"] == "test"].copy()

    dev_df = df[df["split"].isin(["train", "val"])].copy()

    if dev_df.empty:
        raise ValueError("No training/validation data found.")

    dev_df["patient_id"] = dev_df["filename"].apply(
        extract_patient_id
    )

    test_df["patient_id"] = test_df["filename"].apply(
        extract_patient_id
    )

    if dev_df["label"].nunique() < 2:
        raise ValueError(
            "Training/validation data must contain both classes."
        )

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    train_indices, val_indices = next(
        splitter.split(
            dev_df,
            dev_df["label"],
            groups=dev_df["patient_id"]
        )
    )

    train_df = dev_df.iloc[train_indices].copy()
    val_df = dev_df.iloc[val_indices].copy()

    # Check patient leakage between train and validation.
    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])

    if train_patients.intersection(val_patients):
        raise ValueError(
            "Patient leakage detected between train and validation."
        )

    # Check exact file-hash leakage across all splits.
    train_hashes = set(train_df["hash"])
    val_hashes = set(val_df["hash"])
    test_hashes = set(test_df["hash"])

    if train_hashes.intersection(val_hashes):
        raise ValueError("Duplicate images found in train and validation.")

    if train_hashes.intersection(test_hashes):
        raise ValueError("Duplicate images found in train and test.")

    if val_hashes.intersection(test_hashes):
        raise ValueError("Duplicate images found in validation and test.")

    # Report patient IDs shared with test for manual review.
    shared_test_patients = (
        train_patients.union(val_patients)
    ).intersection(set(test_df["patient_id"]))

    if shared_test_patients:
        print(
            "WARNING: Patient IDs shared with test:",
            len(shared_test_patients)
        )

    print("Train images:", len(train_df))
    print("Validation images:", len(val_df))
    print("Test images:", len(test_df))

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


def get_class_weights(train_df):
    """
    Calculate class weights using training data only.
    """

    classes = np.array([0, 1])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].values
    )

    return {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }


def preprocess_image(image_path, image_size=224):
    """
    Read image, convert to grayscale, resize and normalize.
    Output shape: (224, 224, 1)
    """

    image = Image.open(image_path).convert("L")

    image = image.resize(
        (image_size, image_size)
    )

    image = np.array(
        image,
        dtype=np.float32
    )

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=-1
    )

    return image


def process_split(df, image_size=224):
    """
    Convert a dataframe split into image and label arrays.
    """

    images = []
    labels = []

    for _, row in df.iterrows():

        image = preprocess_image(
            row["path"],
            image_size=image_size
        )

        images.append(image)
        labels.append(row["label"])

    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)

    return X, y


def build_augmentation():
    """
    Data augmentation layers.
    Applied only to training data.
    """

    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),
        tf.keras.layers.RandomBrightness(0.1)
    ], name="data_augmentation")


def make_dataset(
    X,
    y,
    training=False,
    augment=True,
    batch_size=32
):
    """
    Create a TensorFlow dataset.
    Augmentation is applied only during training.
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (X, y)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(X),
            reshuffle_each_iteration=True
        )

    if training and augment:
        augmentation = build_augmentation()

        def apply_augmentation(image, label):
            image = augmentation(
                image,
                training=True
            )
            return image, label

        dataset = dataset.map(
            apply_augmentation,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


def load_datasets(config):
    """
    Load, clean, split and preprocess the dataset.
    Returns train and validation TensorFlow datasets.
    """

    dataset_path = config["paths"]["dataset_dir"]

    image_size = config["dataset"].get(
        "image_size", 224
    )

    batch_size = config["training"].get(
        "batch_size", 32
    )

    seed = config["training"].get(
        "seed", 42
    )

    train_df, val_df, test_df = split_data(
        clean_data(
            load_records(dataset_path)
        ),
        seed=seed
    )

    X_train, y_train = process_split(
        train_df, image_size
    )

    X_val, y_val = process_split(
        val_df, image_size
    )

    class_weights = get_class_weights(train_df)

    train_ds = make_dataset(
        X_train,
        y_train,
        training=True,
        augment=True,
        batch_size=batch_size
    )

    val_ds = make_dataset(
        X_val,
        y_val,
        training=False,
        augment=False,
        batch_size=batch_size
    )

    print("Class weights:", class_weights)

    return train_ds, val_ds
