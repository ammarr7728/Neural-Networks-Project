import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
import pandas as pd

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
    config = load_config(config_path)

    seed = config["training"]["seed"]
    set_seed(seed)

    Path(config["paths"]["model_dir"]).mkdir(
        parents=True, exist_ok=True
    )
    Path(config["paths"]["experiment_dir"]).mkdir(
        parents=True, exist_ok=True
    )

    train_ds, val_ds = load_datasets(config)
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

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=config["paths"]["best_model"],
        monitor="val_recall",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_recall",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["epochs"],
        callbacks=[checkpoint, early_stopping],
    )

    history_path = (
        Path(config["paths"]["experiment_dir"])
        / "training_history.csv"
    )

    pd.DataFrame(history.history).to_csv(
        history_path, index=False
    )

    print("Training completed successfully.")
    print(f"Best model saved at: {config['paths']['best_model']}")
    print(f"Training history saved at: {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Chest X-Ray classification model"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the project configuration file",
    )

    args = parser.parse_args()
    train(args.config)
