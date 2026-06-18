# train_all_models.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Basic parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# Training schedule
EPOCHS_FROZEN = 8
EPOCHS_FINETUNE = 12
LR_FROZEN = 1e-3
LR_FINETUNE = 1e-5

# Directory containing class folders
data_dir = '.'

# Class definitions
class_names = ['Foam-Heavy', 'Foam-mild', 'Post-Antifoam Addition', 'Foam-Medium', 'No Foam']
num_classes = len(class_names)

print(f"Classes: {class_names}")
print(f"Number of classes: {num_classes}")


def load_data():
    """
    Load only original images from each class folder.
    Skips augmented subfolders to reduce leakage.
    """
    all_images = []
    all_labels = []

    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            continue

        class_count = 0

        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)

            if os.path.isdir(img_path):
                continue

            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img = keras.preprocessing.image.load_img(
                        img_path,
                        target_size=(IMG_HEIGHT, IMG_WIDTH)
                    )
                    img_array = keras.preprocessing.image.img_to_array(img)
                    all_images.append(img_array)
                    all_labels.append(i)
                    class_count += 1
                except Exception as e:
                    print(f"Error loading {img_path}: {str(e)}")

        print(f"Class {class_name}: {class_count} original images loaded")

    return np.array(all_images), np.array(all_labels)


def create_resnet50_model(input_shape=(224, 224, 3), num_classes=5):
    """
    Functional API model to avoid loading issues with nested Sequential + ResNet50.
    Returns:
      - model
      - base_model
    """
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model, base_model


def preprocess_for_resnet50(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    return resnet50_preprocess(X)


def plot_confusion_matrix(cm, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('ResNet50 Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_training_curves(history1, history2, filename):
    plt.figure(figsize=(12, 6))

    train_acc = history1.history.get('accuracy', []) + history2.history.get('accuracy', [])
    val_acc = history1.history.get('val_accuracy', []) + history2.history.get('val_accuracy', [])

    plt.plot(train_acc, marker='o', label='Train Accuracy')
    plt.plot(val_acc, marker='s', linestyle='--', label='Val Accuracy')

    plt.title('ResNet50 Training / Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def train_resnet50(X_train_raw, y_train_cat, X_val_raw, y_val_cat, X_test_raw, y_test_cat, y_train_int):
    print("\nTraining resnet50_model ...")

    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    X_train = preprocess_for_resnet50(X_train_raw)
    X_val = preprocess_for_resnet50(X_val_raw)
    X_test = preprocess_for_resnet50(X_test_raw)

    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_int),
        y=y_train_int
    )
    class_weights = {i: float(w) for i, w in enumerate(class_weights_array)}
    print("Class weights:", class_weights)

    model, base_model = create_resnet50_model(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        num_classes=num_classes
    )

    model.summary()

    # Phase 1
    print("\n===== Phase 1: Train classifier head only =====")
    base_model.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FROZEN),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint_phase1 = tf.keras.callbacks.ModelCheckpoint(
        "models/resnet50_model_phase1_best.keras",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping_phase1 = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    history1 = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS_FROZEN,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[checkpoint_phase1, early_stopping_phase1],
        verbose=1
    )

    # Phase 2
    print("\n===== Phase 2: Fine-tune top ResNet50 layers =====")
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    trainable_count = sum(int(layer.trainable) for layer in base_model.layers)
    print(f"ResNet50 trainable layers after unfreezing: {trainable_count}/{len(base_model.layers)}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FINETUNE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint_phase2 = tf.keras.callbacks.ModelCheckpoint(
        "models/resnet50_model_best.keras",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping_phase2 = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )

    history2 = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS_FINETUNE,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[checkpoint_phase2, early_stopping_phase2, reduce_lr],
        verbose=1
    )

    print("\n===== Final Evaluation on Test Set =====")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=1)
    print(f"resnet50_model - Test accuracy: {test_accuracy:.4f}")

    predictions = model.predict(X_test, verbose=1)
    if isinstance(predictions, list):
        predictions = predictions[0]

    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)

    print("\nClassification Report:")
    print(report)

    plot_confusion_matrix(cm, 'figures/resnet50_model_confusion_matrix.png')
    plot_training_curves(history1, history2, 'figures/resnet50_model_accuracy_curve.png')

    final_keras_path = "models/resnet50_model.keras"
    final_h5_path = "models/resnet50_model.h5"

    model.save(final_keras_path)
    print(f"Model saved to {final_keras_path}")

    try:
        model.save(final_h5_path)
        print(f"Model saved to {final_h5_path}")
    except Exception as e:
        print(f"Warning: could not save .h5 model: {str(e)}")
        try:
            model.save_weights(final_h5_path)
            print(f"Saved weights instead to {final_h5_path}")
        except Exception as e2:
            print(f"Warning: also failed to save weights: {str(e2)}")

    return {
        'model': model,
        'history_phase1': history1.history,
        'history_phase2': history2.history,
        'accuracy': test_accuracy,
        'report': report,
        'cm': cm
    }


def main():
    print(f"TensorFlow version: {tf.__version__}")
    print("Loading and preprocessing data...")

    X_raw, y = load_data()

    print(f"Total images: {len(X_raw)}")
    print(f"Images per class: {Counter(y)}")

    if len(X_raw) == 0:
        print("ERROR: No images found. Check your class folders and image extensions.")
        return

    X_temp, X_test_raw, y_temp, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.15,
        stratify=y,
        random_state=42
    )

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.1765,
        stratify=y_temp,
        random_state=42
    )

    print(f"Training samples: {len(X_train_raw)}")
    print(f"Validation samples: {len(X_val_raw)}")
    print(f"Testing samples: {len(X_test_raw)}")

    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)

    results = {}
    try:
        results["resnet50_model"] = train_resnet50(
            X_train_raw,
            y_train_cat,
            X_val_raw,
            y_val_cat,
            X_test_raw,
            y_test_cat,
            y_train
        )
    except Exception as e:
        print(f"Error training resnet50_model: {str(e)}")
        return

    print("\n===== TRAINING RESULTS =====")
    for model_key, result in results.items():
        print(f"{model_key}: Test Accuracy = {result['accuracy']:.4f}")

    print("\nTraining complete! Model saved to 'models/' and figures to 'figures/'.")


if __name__ == "__main__":
    main()