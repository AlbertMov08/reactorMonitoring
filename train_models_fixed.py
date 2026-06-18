# train_models_fixed.py
# Trains: custom CNN + MobileNetV2 + VGG16 + ResNet50
# Fixes for your case:
# 1) Correct preprocess_input per backbone (inside the model)
# 2) Two-phase training: frozen head -> fine-tune top layers
# 3) Apple Silicon stability: smaller batch for VGG/ResNet + clear_session + gc between models
# 4) Saves files in names your cnn_inference.py expects:
#    models/custom_cnn_model.h5 (full model)
#    models/mobilenetv2_model.h5 (weights)
#    models/vgg16_model.h5 (weights)
#    models/resnet50_model.h5 (weights)
#    plus .keras full models for convenience

import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# -------------------------
# Reproducibility
# -------------------------
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------
# Basic parameters
# -------------------------
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Keep MobileNet/custom comfortable; VGG/ResNet will override to smaller
BATCH_SIZE_DEFAULT = 32
BATCH_SIZE_HEAVY = 4  # Apple Silicon stability for VGG16/ResNet50 (try 8 if you want, 4 is safest)

# Phase 1 (frozen)
EPOCHS_FROZEN = 6
LR_FROZEN = 1e-3

# Phase 2 (fine-tune)
EPOCHS_FINETUNE = 8
LR_FINETUNE = 1e-5

data_dir = "."
class_names = ['Foam-Heavy', 'Foam-mild', 'Post-Antifoam Addition', 'Foam-Medium', 'No Foam']
num_classes = len(class_names)

os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)

print("TensorFlow:", tf.__version__)
print("Classes:", class_names)

# -------------------------
# Data loading
# -------------------------
def load_data(include_augmented=True):
    """
    Loads images into RAM as float32 0..255 arrays.
    NOTE: We do NOT divide by 255 here because pretrained models will preprocess inside the model.
          We'll only /255 for the custom CNN during training.
    """
    all_images = []
    all_labels = []

    def load_from_dir(d, label_idx):
        if not os.path.exists(d):
            return 0
        c = 0
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                p = os.path.join(d, f)
                try:
                    img = keras.preprocessing.image.load_img(p, target_size=(IMG_HEIGHT, IMG_WIDTH))
                    arr = keras.preprocessing.image.img_to_array(img)  # float32 0..255
                    all_images.append(arr)
                    all_labels.append(label_idx)
                    c += 1
                except Exception as e:
                    print("Error loading", p, e)
        return c

    for i, cname in enumerate(class_names):
        class_dir = os.path.join(data_dir, cname)
        if not os.path.exists(class_dir):
            print("Missing dir:", class_dir)
            continue

        n_main = load_from_dir(class_dir, i)

        n_aug = 0
        if include_augmented:
            aug_dir = os.path.join(class_dir, "augmented")
            n_aug = load_from_dir(aug_dir, i)

        print(f"{cname}: {n_main} main + {n_aug} augmented")

    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    print("Total images:", len(X))
    print("Per-class:", Counter(y))
    return X, y


# -------------------------
# Models
# -------------------------
def make_custom_cnn(input_shape):
    # Custom CNN expects input in 0..1 (we'll divide by 255 during training/inference for this model)
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ], name="custom_cnn_model")
    return model

def make_transfer_model(backbone_name, input_shape):
    """
    Creates model with preprocess_input baked in (Lambda layer).
    This is key: VGG/ResNet/MobileNet each want different preprocessing.
    Input should remain 0..255 float32.
    """
    if backbone_name == "vgg16":
        base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
        pre = vgg_pre
    elif backbone_name == "resnet50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
        pre = res_pre
    elif backbone_name == "mobilenetv2":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
        pre = mob_pre
    else:
        raise ValueError("Unknown backbone: " + backbone_name)

    base.trainable = False

    inp = layers.Input(shape=input_shape)
    x = layers.Lambda(pre, name=f"{backbone_name}_preprocess")(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inp, out, name=f"{backbone_name}_model")
    return model, base

def freeze_batchnorm_layers(m):
    for layer in m.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

def fine_tune_backbone(backbone_name, base_model):
    """
    Carefully unfreeze only a top portion to avoid instability + reduce memory usage.
    """
    base_model.trainable = True

    if backbone_name == "vgg16":
        # Unfreeze block4 and block5 only
        for layer in base_model.layers:
            layer.trainable = ("block4" in layer.name) or ("block5" in layer.name)

    elif backbone_name == "resnet50":
        # Unfreeze last ~50 layers; keep BN frozen
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        for layer in base_model.layers[-50:]:
            if "batch_normalization" in layer.name.lower():
                layer.trainable = False
            else:
                layer.trainable = True
        freeze_batchnorm_layers(base_model)

    elif backbone_name == "mobilenetv2":
        # Unfreeze last ~40 layers; keep BN frozen
        for layer in base_model.layers[:-40]:
            layer.trainable = False
        for layer in base_model.layers[-40:]:
            if "batch_normalization" in layer.name.lower():
                layer.trainable = False
            else:
                layer.trainable = True
        freeze_batchnorm_layers(base_model)

def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

def save_for_inference(model, model_key):
    """
    Saves both:
      - full model: models/<model_key>.keras
      - weights only: models/<model_key>.h5   (matches your cnn_inference.py load_weights)
    """
    keras_path = os.path.join("models", f"{model_key}.keras")
    weights_path = os.path.join("models", f"{model_key}.h5")

    model.save(keras_path)
    model.save_weights(weights_path)

    print("Saved full model:", keras_path)
    print("Saved weights:", weights_path)

def clear_mem():
    tf.keras.backend.clear_session()
    gc.collect()


# -------------------------
# Training
# -------------------------
def main():
    X, y = load_data(include_augmented=True)

    # Split
    X_train, X_test, y_train_idx, y_test_idx = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # One-hot
    y_train = keras.utils.to_categorical(y_train_idx, num_classes)
    y_test = keras.utils.to_categorical(y_test_idx, num_classes)

    # Class weights (helps imbalances/noise)
    cw = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_idx), y=y_train_idx)
    class_weight = {i: float(w) for i, w in zip(np.unique(y_train_idx), cw)}
    print("class_weight:", class_weight)

    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

    # -------------------------
    # 1) Custom CNN (single-phase; uses /255.0)
    # -------------------------
    print("\n==============================")
    print("Training custom_cnn_model")
    print("==============================")
    clear_mem()

    custom = make_custom_cnn(input_shape)
    compile_model(custom, LR_FROZEN)

    X_train_custom = X_train / 255.0
    X_test_custom = X_test / 255.0

    custom.fit(
        X_train_custom, y_train,
        validation_data=(X_test_custom, y_test),
        epochs=(EPOCHS_FROZEN + EPOCHS_FINETUNE),
        batch_size=BATCH_SIZE_DEFAULT,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        ],
        verbose=1
    )

    # Your inference loads this with load_model("models/custom_cnn_model.h5")
    custom.save(os.path.join("models", "custom_cnn_model.h5"))
    print("Saved: models/custom_cnn_model.h5")

    del custom
    clear_mem()

    # -------------------------
    # 2) Transfer models (two-phase)
    # -------------------------
    for backbone in ["mobilenetv2", "vgg16", "resnet50"]:
        model_name_print = backbone.upper()
        print("\n==============================")
        print(f"Training {model_name_print}")
        print("==============================")

        clear_mem()

        # Apple Silicon stability: smaller batch for heavy models
        bs = BATCH_SIZE_DEFAULT
        if backbone in ["vgg16", "resnet50"]:
            bs = BATCH_SIZE_HEAVY

        # Build
        model, base = make_transfer_model(backbone, input_shape)

        # ---- Phase 1: frozen head ----
        print(f"\n[{model_name_print}] Phase 1 (frozen) | batch_size={bs}")
        compile_model(model, LR_FROZEN)

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS_FROZEN,
            batch_size=bs,
            class_weight=class_weight,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
            ],
            verbose=1
        )

        # ---- Phase 2: fine-tune ----
        print(f"\n[{model_name_print}] Phase 2 (fine-tune) | batch_size={bs}")
        fine_tune_backbone(backbone, base)
        compile_model(model, LR_FINETUNE)

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS_FINETUNE,
            batch_size=bs,
            class_weight=class_weight,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
            ],
            verbose=1
        )

        # Save in the exact keys your cnn_inference.py expects
        # (it looks for models/mobilenetv2_model.h5 etc and calls load_weights)
        model_key = f"{backbone}_model"
        save_for_inference(model, model_key)

        del model
        del base
        clear_mem()

    print("\nDone! Models saved in ./models")


if __name__ == "__main__":
    main()
