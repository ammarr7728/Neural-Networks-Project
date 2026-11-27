
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.data import load_datasets
from src.models import build_model


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def train(config_path):
    # Load configuration
    config = load_config(config_path)

    seed = config["training"]["seed"]
    set_seed(seed)

    # Create output directories
    model_dir = Path(config["paths"]["model_dir"])
    experiment_dir = Path(config["paths"]["experiment_dir"])

    model_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets and class weights
    train_ds, val_ds, class_weights = load_datasets(config)

    # Build model
    model = build_model(config)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["training"]["learning_rate"]
        ),
        loss=config["model"]["loss"],
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    model.summary()

    # Save the best model based on validation recall
    best_model_path = Path(config["paths"]["best_model"])
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_recall",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_recall",
        mode="max",
        patience=config["training"].get("patience", 5),
        restore_best_weights=True,
        verbose=1,
    )

    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["epochs"],
        class_weight=class_weights,
        callbacks=[checkpoint, early_stopping],
    )

    # Save training history
    history_path = experiment_dir / f"training_history_seed_{seed}.csv"

    pd.DataFrame(history.history).to_csv(
        history_path,
        index=False
    )

    print("\nTraining completed successfully!")
    print("Best model saved at:", best_model_path)
    print("Training history saved at:", history_path)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Chest X-Ray Pneumonia Classifier"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml"
    )

    args = parser.parse_args()

    train(args.config)
