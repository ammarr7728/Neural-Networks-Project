"""Compatible model definitions for the team's Chest X-Ray pipeline."""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="ChestXRay")
class VGG16ImageNetPreprocess(tf.keras.layers.Layer):
    """Accept grayscale float images [0,1]; create VGG16 ImageNet BGR inputs."""
    def call(self, inputs):
        x = tf.concat([inputs, inputs, inputs], axis=-1) * 255.0
        x = tf.reverse(x, axis=[-1])
        mean = tf.constant([103.939, 116.779, 123.68], dtype=x.dtype)
        return x - mean


def build_cnn(filters=32, dropout=0.5, activation="relu"):
    """Model A. Original team function signature and architecture retained."""
    inputs = tf.keras.Input(shape=(224, 224, 1), name="xray_grayscale")
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation=activation)(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(filters * 2, 3, padding="same", activation=activation)(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(filters * 4, 3, padding="same", activation=activation)(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation=activation)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="custom_cnn")


def _build_vgg16(finetune=False, dropout=0.5, l2_strength=1e-4,
                 dense_units=256, weights="imagenet"):
    inputs = tf.keras.Input(shape=(224, 224, 1), name="xray_grayscale")
    x = VGG16ImageNetPreprocess(name="vgg16_imagenet_preprocessing")(inputs)
    backbone = tf.keras.applications.VGG16(
        include_top=False, weights=weights, input_shape=(224, 224, 3)
    )
    if finetune:
        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = layer.name.startswith("block5_")
    else:
        backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        dense_units, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
    )(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    name = "vgg16_finetuned" if finetune else "vgg16_frozen"
    return tf.keras.Model(inputs, outputs, name=name)


def build_vgg16_frozen(dropout=0.5, l2_strength=1e-4,
                       dense_units=256, weights="imagenet"):
    """Model B: frozen ImageNet VGG16 feature extractor."""
    return _build_vgg16(False, dropout, l2_strength, dense_units, weights)


def build_vgg16_finetuned(dropout=0.5, l2_strength=1e-4,
                          dense_units=256, weights="imagenet"):
    """Model C: fine-tune VGG16 block5; all earlier blocks remain frozen."""
    return _build_vgg16(True, dropout, l2_strength, dense_units, weights)


def build_model(config):
    """Backward-compatible factory used by the existing src/train.py."""
    cfg = config.get("model", {})
    name = cfg.get("name", "custom_cnn").lower()
    dropout = float(cfg.get("dropout", 0.5))
    if name in ("custom_cnn", "cnn", "model_a"):
        return build_cnn(
            filters=int(cfg.get("filters", 32)),
            dropout=dropout,
            activation=cfg.get("activation", "relu")
        )
    common = dict(
        dropout=dropout,
        l2_strength=float(cfg.get("l2_strength", 1e-4)),
        dense_units=int(cfg.get("dense_units", 256)),
        weights=cfg.get("pretrained_weights", "imagenet")
    )
    if name in ("vgg16_frozen", "model_b"):
        return build_vgg16_frozen(**common)
    if name in ("vgg16_finetuned", "model_c"):
        return build_vgg16_finetuned(**common)
    raise ValueError(
        f"Unknown model.name={name!r}; use custom_cnn, vgg16_frozen, or vgg16_finetuned"
    )

