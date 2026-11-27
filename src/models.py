
import tensorflow as tf


def build_cnn(
    filters=32,
    dropout=0.5,
    activation="relu"
):
    inputs = tf.keras.Input(shape=(224, 224, 1))

    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", activation=activation
    )(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(
        filters * 2, 3, padding="same", activation=activation
    )(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(
        filters * 4, 3, padding="same", activation=activation
    )(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(64, activation=activation)(x)

    x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid"
    )(x)

    return tf.keras.Model(inputs, outputs)


def build_model(config):
    model_config = config.get("model", {})

    return build_cnn(
        filters=model_config.get("filters", 32),
        dropout=model_config.get("dropout", 0.5),
        activation=model_config.get("activation", "relu")
    )
