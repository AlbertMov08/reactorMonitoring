# train_models_fixed.py
# Main Foam model training file
#
# Trains exactly these 4 models:
# 1) Custom CNN
# 2) MobileNetV2
# 3) ResNet50
# 4) EfficientNetB0
#
# VGG16 removed completely.
#
# Saves models as:
# models/custom_cnn_model.keras
# models/mobilenetv2_model.keras
# models/resnet50_model.keras
# models/efficientnetb0_model.keras
#
# Also saves weights using the required Keras 3 naming format:
# models/custom_cnn_model.weights.h5
# models/mobilenetv2_model.weights.h5
# models/resnet50_model.weights.h5
# models/efficientnetb0_model.weights.h5
#
# IMPORTANT:
# Keras 3 requires save_weights() files to end with `.weights.h5`.
# Do NOT use model.save_weights("models/model_name.h5").

import os
import gc
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------
# Reproducibility
# -------------------------
np.random.seed(42)
tf.random.set_seed(42)


# -------------------------
# Project config
# -------------------------
DATA_DIR = "."

IMG_HEIGHT = 224
IMG_WIDTH = 224
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

CLASS_NAMES = [
    "Foam-Heavy",
    "Foam-mild",
    "Post-Antifoam Addition",
    "Foam-Medium",
    "No Foam",
]

NUM_CLASSES = len(CLASS_NAMES)

USE_AUGMENTED = True

# Phase 1: train only classification head while base is frozen
EPOCHS_FROZEN = 8
LR_FROZEN = 1e-3

# Phase 2: fine-tune top layers of pretrained base model
EPOCHS_FINETUNE = 10
LR_FINETUNE = 1e-5

# MobileNet and Custom CNN can usually use larger batches
BATCH_SIZE_LIGHT = 32

# ResNet50 and EfficientNetB0 are heavier, safer on Mac with smaller batch
BATCH_SIZE_HEAVY = 4

os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)

print("RUNNING FILE:", os.path.abspath(__file__))
print("TensorFlow version:", tf.__version__)
print("Classes:", CLASS_NAMES)
print("Number of classes:", NUM_CLASSES)


# -------------------------
# Utility helpers
# -------------------------
def clear_memory():
    tf.keras.backend.clear_session()
    gc.collect()


def load_images_from_folder(folder_path, label_idx):
    images = []
    labels = []

    if not os.path.exists(folder_path):
        return images, labels

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isdir(file_path):
            continue

        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                img = keras.preprocessing.image.load_img(
                    file_path,
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                )
                img_array = keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(label_idx)
            except Exception as e:
                print(f"Could not load image: {file_path}")
                print("Error:", e)

    return images, labels


def load_data():
    all_images = []
    all_labels = []

    print("\nLoading image data...")

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATA_DIR, class_name)

        if not os.path.exists(class_dir):
            print(f"WARNING: Missing class folder: {class_dir}")
            continue

        main_images, main_labels = load_images_from_folder(class_dir, label_idx)

        aug_images = []
        aug_labels = []

        if USE_AUGMENTED:
            augmented_dir = os.path.join(class_dir, "augmented")
            aug_images, aug_labels = load_images_from_folder(augmented_dir, label_idx)

        all_images.extend(main_images)
        all_labels.extend(main_labels)
        all_images.extend(aug_images)
        all_labels.extend(aug_labels)

        print(
            f"{class_name}: "
            f"{len(main_images)} main + {len(aug_images)} augmented = "
            f"{len(main_images) + len(aug_images)}"
        )

    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    print("\nTotal images:", len(X))
    print("Images per class:", Counter(y))

    return X, y


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def get_callbacks(model_key):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=f"models/{model_key}_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
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
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def save_model_outputs(model, model_key):
    """
    Saves the model safely for Keras 3.

    Full model:
        models/<model_key>.keras

    Weights only:
        models/<model_key>.weights.h5

    The `.weights.h5` ending is required when using model.save_weights().
    """
    full_model_path = os.path.join("models", f"{model_key}.keras")
    weights_path = os.path.join("models", f"{model_key}.weights.h5")

    model.save(full_model_path)
    model.save_weights(weights_path)

    print(f"Saved full model: {full_model_path}")
    print(f"Saved weights: {weights_path}")


def plot_training_history(histories, model_key):
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []

    for history in histories:
        train_acc.extend(history.history.get("accuracy", []))
        val_acc.extend(history.history.get("val_accuracy", []))
        train_loss.extend(history.history.get("loss", []))
        val_loss.extend(history.history.get("val_loss", []))

    if len(train_acc) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(train_acc, marker="o", label="Train Accuracy")
        plt.plot(val_acc, marker="s", label="Validation Accuracy")
        plt.title(f"{model_key} Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"figures/{model_key}_accuracy_curve.png")
        plt.close()

    if len(train_loss) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, marker="o", label="Train Loss")
        plt.plot(val_loss, marker="s", label="Validation Loss")
        plt.title(f"{model_key} Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"figures/{model_key}_loss_curve.png")
        plt.close()


def evaluate_model(model, model_key, X_test, y_test_cat, y_test_idx):
    print(f"\nEvaluating {model_key}...")

    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=1)

    print(f"{model_key} test loss: {test_loss:.4f}")
    print(f"{model_key} test accuracy: {test_accuracy:.4f}")

    predictions = model.predict(X_test, verbose=1)
    y_pred = np.argmax(predictions, axis=1)

    report = classification_report(
        y_test_idx,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test_idx,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    print("\nClassification Report:")
    print(report)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title(f"{model_key} Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(f"figures/{model_key}_confusion_matrix.png")
    plt.close()

    return test_accuracy


# -------------------------
# Model builders
# -------------------------
def build_mobilenetv2_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE,
    )
    base_model.trainable = False

    model = keras.Sequential(
        [
            layers.Input(shape=INPUT_SHAPE),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="mobilenetv2_model",
    )

    return model, base_model


def build_resnet50_model():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE,
    )
    base_model.trainable = False

    inputs = layers.Input(shape=INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="resnet50_model")

    return model, base_model


def build_efficientnetb0_model():
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE,
    )
    base_model.trainable = False

    model = keras.Sequential(
        [
            layers.Input(shape=INPUT_SHAPE),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="efficientnetb0_model",
    )

    return model, base_model


def build_custom_cnn_model():
    model = keras.Sequential(
        [
            layers.Input(shape=INPUT_SHAPE),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="custom_cnn_model",
    )

    return model


# -------------------------
# Fine-tuning helpers
# -------------------------
def freeze_batchnorm_layers(model):
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def fine_tune_base_model(model_key, base_model):
    base_model.trainable = True

    if model_key == "mobilenetv2_model":
        layers_to_unfreeze = 40
    elif model_key == "resnet50_model":
        layers_to_unfreeze = 50
    elif model_key == "efficientnetb0_model":
        layers_to_unfreeze = 50
    else:
        layers_to_unfreeze = 0

    if layers_to_unfreeze > 0:
        for layer in base_model.layers[:-layers_to_unfreeze]:
            layer.trainable = False

        for layer in base_model.layers[-layers_to_unfreeze:]:
            layer.trainable = True

    freeze_batchnorm_layers(base_model)

    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    print(
        f"{model_key}: trainable base layers = "
        f"{trainable_count}/{len(base_model.layers)}"
    )


# -------------------------
# Training functions
# -------------------------
def train_transfer_model(
    model_key,
    display_name,
    builder,
    preprocess_func,
    batch_size,
    X_train_raw,
    X_val_raw,
    X_test_raw,
    y_train_cat,
    y_val_cat,
    y_test_cat,
    y_test_idx,
    class_weight,
):
    print("\n==============================")
    print(f"Training {display_name}")
    print("==============================")

    clear_memory()

    model, base_model = builder()

    # Preprocess outside the model. This keeps each backbone using its correct preprocessing.
    X_train = preprocess_func(X_train_raw.copy())
    X_val = preprocess_func(X_val_raw.copy())
    X_test = preprocess_func(X_test_raw.copy())

    histories = []

    print(f"\n{display_name} Phase 1: frozen base")
    base_model.trainable = False
    compile_model(model, LR_FROZEN)

    history1 = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS_FROZEN,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=get_callbacks(model_key),
        verbose=1,
    )

    histories.append(history1)

    print(f"\n{display_name} Phase 2: fine-tuning")
    fine_tune_base_model(model_key, base_model)
    compile_model(model, LR_FINETUNE)

    history2 = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS_FINETUNE,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=get_callbacks(model_key),
        verbose=1,
    )

    histories.append(history2)

    test_accuracy = evaluate_model(
        model=model,
        model_key=model_key,
        X_test=X_test,
        y_test_cat=y_test_cat,
        y_test_idx=y_test_idx,
    )

    plot_training_history(histories, model_key)
    save_model_outputs(model, model_key)

    del model
    del base_model
    clear_memory()

    return test_accuracy


def train_custom_cnn(
    X_train_raw,
    X_val_raw,
    X_test_raw,
    y_train_cat,
    y_val_cat,
    y_test_cat,
    y_test_idx,
    class_weight,
):
    model_key = "custom_cnn_model"

    print("\n==============================")
    print("Training Custom CNN")
    print("==============================")

    clear_memory()

    model = build_custom_cnn_model()

    X_train = X_train_raw.astype(np.float32) / 255.0
    X_val = X_val_raw.astype(np.float32) / 255.0
    X_test = X_test_raw.astype(np.float32) / 255.0

    compile_model(model, LR_FROZEN)

    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS_FROZEN + EPOCHS_FINETUNE,
        batch_size=BATCH_SIZE_LIGHT,
        class_weight=class_weight,
        callbacks=get_callbacks(model_key),
        verbose=1,
    )

    test_accuracy = evaluate_model(
        model=model,
        model_key=model_key,
        X_test=X_test,
        y_test_cat=y_test_cat,
        y_test_idx=y_test_idx,
    )

    plot_training_history([history], model_key)
    save_model_outputs(model, model_key)

    del model
    clear_memory()

    return test_accuracy


# -------------------------
# Main
# -------------------------
def main():
    X, y = load_data()

    if len(X) == 0:
        print("ERROR: No images found.")
        print("Make sure these folders exist:")
        for class_name in CLASS_NAMES:
            print("-", class_name)
        return

    print("\nSplitting data...")

    # 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.1765,
        stratify=y_temp,
        random_state=42,
    )

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples:", len(X_test))

    y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES)
    y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )

    class_weight = {
        int(cls): float(weight)
        for cls, weight in zip(np.unique(y_train), class_weights_array)
    }

    print("Class weights:", class_weight)

    results = {}

    # 1. Custom CNN
    results["custom_cnn_model"] = train_custom_cnn(
        X_train_raw=X_train,
        X_val_raw=X_val,
        X_test_raw=X_test,
        y_train_cat=y_train_cat,
        y_val_cat=y_val_cat,
        y_test_cat=y_test_cat,
        y_test_idx=y_test,
        class_weight=class_weight,
    )

    # 2. MobileNetV2
    results["mobilenetv2_model"] = train_transfer_model(
        model_key="mobilenetv2_model",
        display_name="MobileNetV2",
        builder=build_mobilenetv2_model,
        preprocess_func=mobilenet_preprocess,
        batch_size=BATCH_SIZE_LIGHT,
        X_train_raw=X_train,
        X_val_raw=X_val,
        X_test_raw=X_test,
        y_train_cat=y_train_cat,
        y_val_cat=y_val_cat,
        y_test_cat=y_test_cat,
        y_test_idx=y_test,
        class_weight=class_weight,
    )

    # 3. ResNet50
    results["resnet50_model"] = train_transfer_model(
        model_key="resnet50_model",
        display_name="ResNet50",
        builder=build_resnet50_model,
        preprocess_func=resnet50_preprocess,
        batch_size=BATCH_SIZE_HEAVY,
        X_train_raw=X_train,
        X_val_raw=X_val,
        X_test_raw=X_test,
        y_train_cat=y_train_cat,
        y_val_cat=y_val_cat,
        y_test_cat=y_test_cat,
        y_test_idx=y_test,
        class_weight=class_weight,
    )

    # 4. EfficientNetB0
    results["efficientnetb0_model"] = train_transfer_model(
        model_key="efficientnetb0_model",
        display_name="EfficientNetB0",
        builder=build_efficientnetb0_model,
        preprocess_func=efficientnet_preprocess,
        batch_size=BATCH_SIZE_HEAVY,
        X_train_raw=X_train,
        X_val_raw=X_val,
        X_test_raw=X_test,
        y_train_cat=y_train_cat,
        y_val_cat=y_val_cat,
        y_test_cat=y_test_cat,
        y_test_idx=y_test,
        class_weight=class_weight,
    )

    print("\n==============================")
    print("FINAL TRAINING RESULTS")
    print("==============================")

    for model_key, accuracy in results.items():
        print(f"{model_key}: test accuracy = {accuracy:.4f}")

    print("\nDone.")
    print("Models saved in: ./models")
    print("Figures saved in: ./figures")


if __name__ == "__main__":
    main()

