"""
train_models_fixed.py

High-reliability training pipeline for the reactor foam classifier.

Major fixes compared with the older pipeline
---------------------------------------------
* Uses ORIGINAL images as the source of truth; pre-generated augmented images
  are ignored by default so train/validation/test leakage cannot happen.
* Splits data BEFORE augmentation.
* Supports nested visual subtypes for Foam-Medium and Foam-Heavy while keeping
  the final labels exactly the same five classes.
* Supports capture/video grouping so frames from the same experiment can stay
  in the same split.
* Applies realistic augmentation on-the-fly to TRAINING images only.
* Preserves image aspect ratio with resize-with-padding instead of stretching.
* Balances both classes and rare visual subtypes using sample weights.
* Fine-tunes ImageNet backbones carefully with BatchNorm frozen.
* Uses early stopping and learning-rate reduction; long maximum training is OK.
* Evaluates on untouched original images and writes per-subtype diagnostics.
* Learns ensemble weights from validation predictions instead of hard-coding
  arbitrary model weights.
* Upgrades ONLY the custom CNN to a deeper residual architecture with its own
  optimizer/training settings. MobileNetV2, ResNet50, and EfficientNetB0 are
  unchanged.

Recommended folder layout for the best split quality
----------------------------------------------------
Foam-Heavy/
    white_full_foam/
        video_01/
            frame001.jpg
            frame002.jpg
        video_02/
            ...
    yellow_heavy/
        video_03/
            ...
Foam-Medium/
    yellow_medium/
        video_04/
            ...

The FIRST folder under the class is a visual subtype. The SECOND folder (when
present) is treated as the capture/video/experiment group and is kept together
when splitting. If you do not have a second folder, filenames containing
"frame" are also grouped by their prefix when possible.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # older scikit-learn fallback
    StratifiedGroupKFold = None
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess


# -----------------------------------------------------------------------------
# Reproducibility / configuration
# -----------------------------------------------------------------------------
SEED = 42
keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path(".")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("figures")
REPORTS_DIR = Path("training_reports")
for p in (MODELS_DIR, FIGURES_DIR, REPORTS_DIR):
    p.mkdir(exist_ok=True)

CLASS_NAMES = [
    "Foam-Heavy",
    "Foam-Mild",
    "Post-Antifoam Addition",
    "Foam-Medium",
    "No Foam",
]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUGMENTED_DIR_NAME = "augmented"
IMG_HEIGHT = 224
IMG_WIDTH = 224
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# Intentionally long maxima. EarlyStopping normally ends training sooner.
HEAD_EPOCHS = 20
FINETUNE_EPOCHS = 45
HEAD_LR = 3e-4
FINETUNE_LR = 1e-5
WEIGHT_DECAY = 1e-5
LABEL_SMOOTHING = 0.04
EARLY_STOP_PATIENCE = 8
LR_PATIENCE = 3

BATCH_SIZE_CUSTOM = 16
BATCH_SIZE_TRANSFER = 8

# Custom CNN only. These settings do NOT affect MobileNetV2, ResNet50,
# or EfficientNetB0. The custom network starts from random weights, so it
# needs a lower learning rate and a longer training budget than the
# pretrained transfer-learning models.
CUSTOM_CNN_EPOCHS = 120
CUSTOM_CNN_LR = 2e-4
CUSTOM_CNN_WEIGHT_DECAY = 5e-5
CUSTOM_CNN_LABEL_SMOOTHING = 0.02
CUSTOM_CNN_EARLY_STOP_PATIENCE = 15
CUSTOM_CNN_LR_PATIENCE = 5

# Fine-tune the top fraction of each pretrained backbone.
FINE_TUNE_FRACTION = {
    "mobilenetv2_model": 0.35,
    "resnet50_model": 0.30,
    "efficientnetb0_model": 0.35,
}

# Rare subtype weighting is useful for the two classes with multiple visual
# appearances. It does NOT create new labels.
SUBTYPE_WEIGHT_CLASSES = {"Foam-Medium", "Foam-Heavy"}
MIN_SAMPLE_WEIGHT = 0.40
MAX_SAMPLE_WEIGHT = 3.00

# Evaluation split. Group-aware 6-fold split gives ~16.7% test and ~16.7% val
# when there are enough capture/video groups in every class.
GROUP_SPLIT_FOLDS = 6

# Uncertainty defaults written for inference; inference can still show the most
# likely class but marks low-confidence decisions as uncertain.
INFERENCE_SMOOTHING_WINDOW = 7
INFERENCE_MIN_CONFIDENCE = 0.45
INFERENCE_MIN_MARGIN = 0.06

print("RUNNING FILE:", os.path.abspath(__file__))
print("TensorFlow:", tf.__version__)
print("Classes:", CLASS_NAMES)


# -----------------------------------------------------------------------------
# Dataset discovery
# -----------------------------------------------------------------------------
def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_subtype(class_dir: Path, path: Path) -> str:
    rel_parent = path.parent.relative_to(class_dir)
    if str(rel_parent) == ".":
        return "root"
    return rel_parent.parts[0]


def infer_capture_group(class_dir: Path, path: Path, subtype: str) -> str:
    """
    Best-effort group ID used to stop frames from one capture leaking across
    train/val/test.

    Priority:
      1) <class>/<subtype>/<video_or_experiment>/...  (recommended)
      2) filename prefix before tokens such as _frame_001
      3) unique image path (safe for duplicates, but cannot infer video origin)
    """
    rel_parent = path.parent.relative_to(class_dir)
    parts = rel_parent.parts

    if len(parts) >= 2:
        return f"{subtype}::{parts[1]}"

    stem = path.stem
    patterns = [
        r"^(.*?)[_-](?:frame|frm)[_-]?\d+(?:[_-].*)?$",
        r"^(.*?)[_-](?:sec|second|timestamp)[_-]?\d+(?:[_-].*)?$",
    ]
    for pattern in patterns:
        m = re.match(pattern, stem, flags=re.IGNORECASE)
        if m and m.group(1).strip("_- "):
            return f"{subtype}::{m.group(1).strip('_- ')}"

    return f"{subtype}::{path.as_posix()}"


def discover_original_records() -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    seen_hashes: Dict[str, str] = {}
    duplicate_rows: List[Dict[str, str]] = []

    for class_name in CLASS_NAMES:
        class_dir = DATA_DIR / class_name
        if not class_dir.exists():
            print(f"WARNING: missing class folder: {class_dir}")
            continue

        for root, dirs, files in os.walk(class_dir):
            root_path = Path(root)
            rel_parts = [p.lower() for p in root_path.relative_to(class_dir).parts]
            if AUGMENTED_DIR_NAME.lower() in rel_parts:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d.lower() != AUGMENTED_DIR_NAME.lower()]

            for filename in files:
                path = root_path / filename
                if not is_image(path):
                    continue

                try:
                    digest = sha256_file(path)
                except Exception as e:
                    print(f"Skipping unreadable file {path}: {e}")
                    continue

                # Exact duplicate images add no information and can cause leakage.
                if digest in seen_hashes:
                    duplicate_rows.append(
                        {
                            "duplicate": str(path),
                            "kept": seen_hashes[digest],
                            "sha256": digest,
                        }
                    )
                    continue
                seen_hashes[digest] = str(path)

                subtype = infer_subtype(class_dir, path)
                capture_group = infer_capture_group(class_dir, path, subtype)
                records.append(
                    {
                        "path": str(path),
                        "class_name": class_name,
                        "label": CLASS_TO_INDEX[class_name],
                        "subtype": subtype,
                        # Prefix with class so identical group names in different
                        # labels cannot accidentally be merged.
                        "group": f"{class_name}::{capture_group}",
                        "sha256": digest,
                    }
                )

    if duplicate_rows:
        with (REPORTS_DIR / "exact_duplicates_skipped.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["duplicate", "kept", "sha256"])
            writer.writeheader()
            writer.writerows(duplicate_rows)

    return records


def print_dataset_summary(records: Sequence[Dict[str, object]]) -> None:
    print("\nORIGINAL DATASET SUMMARY")
    print("=" * 72)
    by_class = Counter(r["class_name"] for r in records)
    for class_name in CLASS_NAMES:
        print(f"{class_name}: {by_class[class_name]}")
        if class_name in SUBTYPE_WEIGHT_CLASSES:
            sub = Counter(
                r["subtype"] for r in records if r["class_name"] == class_name
            )
            for subtype, count in sorted(sub.items()):
                print(f"  - {subtype}: {count} real original(s)")

    print("\nImportant: generated augmented/ images are NOT counted here.")
    print("Training augmentation is applied only after the split.")


# -----------------------------------------------------------------------------
# Split logic
# -----------------------------------------------------------------------------
def grouped_split(records: List[Dict[str, object]]):
    """
    Prefer a group-aware split whenever there are at least 3 independent
    capture/video groups per class.

    This dynamically reduces the fold count instead of immediately falling
    back to image-level splitting when fewer than 6 groups are available.
    """
    labels = np.array([int(r["label"]) for r in records])
    groups = np.array([str(r["group"]) for r in records])
    indices = np.arange(len(records))

    def groups_per_class(sub_indices):
        out = {}
        for class_name in CLASS_NAMES:
            class_idx = CLASS_TO_INDEX[class_name]
            ids = [i for i in sub_indices if labels[i] == class_idx]
            out[class_name] = len({groups[i] for i in ids})
        return out

    unique_groups_per_class = groups_per_class(indices)
    min_groups = min(unique_groups_per_class.values()) if unique_groups_per_class else 0

    if StratifiedGroupKFold is not None and min_groups >= 3:
        test_folds = min(GROUP_SPLIT_FOLDS, min_groups)
        print(
            f"\nUsing StratifiedGroupKFold with {test_folds} folds "
            "to keep capture/video groups together."
        )

        sgkf = StratifiedGroupKFold(
            n_splits=test_folds,
            shuffle=True,
            random_state=SEED,
        )
        trainval_idx, test_idx = next(sgkf.split(indices, labels, groups))

        rem_group_counts = groups_per_class(trainval_idx)
        rem_min_groups = min(rem_group_counts.values()) if rem_group_counts else 0

        if rem_min_groups >= 2:
            val_folds = min(max(2, test_folds - 1), rem_min_groups)
            rem_labels = labels[trainval_idx]
            rem_groups = groups[trainval_idx]
            rem_indices = np.arange(len(trainval_idx))

            sgkf_val = StratifiedGroupKFold(
                n_splits=val_folds,
                shuffle=True,
                random_state=SEED + 1,
            )
            train_rel, val_rel = next(
                sgkf_val.split(rem_indices, rem_labels, rem_groups)
            )
            train_idx = trainval_idx[train_rel]
            val_idx = trainval_idx[val_rel]
        else:
            print(
                "WARNING: test split is group-aware, but the remaining data does "
                "not contain enough independent groups for a group-aware validation split."
            )
            train_idx, val_idx = train_test_split(
                trainval_idx,
                test_size=0.20,
                stratify=labels[trainval_idx],
                random_state=SEED + 1,
            )
    else:
        print(
            "\nWARNING: fewer than 3 independent capture/video groups exist "
            "in at least one class."
        )
        print("Groups per class:", unique_groups_per_class)
        print(
            "Falling back to an image-level split. These metrics may be optimistic. "
            "For trustworthy real-world testing, organize screenshots as "
            "<class>/<subtype>/<video_or_experiment>/frame.jpg."
        )
        trainval_idx, test_idx = train_test_split(
            indices,
            test_size=0.16,
            stratify=labels,
            random_state=SEED,
        )
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=0.19,
            stratify=labels[trainval_idx],
            random_state=SEED + 1,
        )

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]
    return train_records, val_records, test_records


def save_split_manifest(
    train_records: Sequence[Dict[str, object]],
    val_records: Sequence[Dict[str, object]],
    test_records: Sequence[Dict[str, object]],
) -> None:
    rows = []
    for split_name, subset in (
        ("train", train_records),
        ("validation", val_records),
        ("test", test_records),
    ):
        for r in subset:
            rows.append({**r, "split": split_name})

    fields = ["split", "class_name", "label", "subtype", "group", "path", "sha256"]
    with (REPORTS_DIR / "dataset_split_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def print_split_summary(name: str, records: Sequence[Dict[str, object]]) -> None:
    counts = Counter(r["class_name"] for r in records)
    print(f"{name}: {len(records)} total -> {dict(counts)}")


# -----------------------------------------------------------------------------
# Sample weighting
# -----------------------------------------------------------------------------
def calculate_train_weights(records: Sequence[Dict[str, object]]) -> np.ndarray:
    class_counts = Counter(r["class_name"] for r in records)
    subtype_counts: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        subtype_counts[str(r["class_name"])][str(r["subtype"])] += 1

    total = len(records)
    weights = []
    for r in records:
        class_name = str(r["class_name"])
        subtype = str(r["subtype"])

        class_weight = total / (NUM_CLASSES * max(1, class_counts[class_name]))
        subtype_weight = 1.0

        if class_name in SUBTYPE_WEIGHT_CLASSES:
            n_subtypes = max(1, len(subtype_counts[class_name]))
            class_n = class_counts[class_name]
            subtype_n = subtype_counts[class_name][subtype]
            subtype_weight = class_n / (n_subtypes * max(1, subtype_n))

        w = class_weight * subtype_weight
        w = float(np.clip(w, MIN_SAMPLE_WEIGHT, MAX_SAMPLE_WEIGHT))
        weights.append(w)

    weights = np.asarray(weights, dtype=np.float32)
    if len(weights):
        weights /= max(float(weights.mean()), 1e-8)
    return weights


# -----------------------------------------------------------------------------
# tf.data pipeline
# -----------------------------------------------------------------------------
GEOMETRIC_AUGMENTER = keras.Sequential(
    [
        # Make position and scale less important so the reactor does not need
        # to appear in the same place every time.
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.035, fill_mode="reflect", seed=SEED + 1),
        layers.RandomTranslation(
            height_factor=0.10,
            width_factor=0.10,
            fill_mode="reflect",
            seed=SEED + 2,
        ),
        layers.RandomZoom(
            height_factor=(-0.10, 0.12),
            width_factor=(-0.10, 0.12),
            fill_mode="reflect",
            seed=SEED + 3,
        ),
    ],
    name="train_geometry_augmentation",
)


def decode_resize(path: tf.Tensor) -> tf.Tensor:
    data = tf.io.read_file(path)
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.cast(image, tf.float32)
    # Preserve full reactor geometry instead of stretching portrait/landscape.
    image = tf.image.resize_with_pad(
        image,
        target_height=IMG_HEIGHT,
        target_width=IMG_WIDTH,
        method="bilinear",
        antialias=True,
    )
    return image


def augment_train(image: tf.Tensor) -> tf.Tensor:
    image = GEOMETRIC_AUGMENTER(image, training=True)
    image = tf.image.random_brightness(image, max_delta=18.0, seed=SEED + 10)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15, seed=SEED + 11)
    image = tf.image.random_saturation(image, lower=0.90, upper=1.10, seed=SEED + 12)
    noise = tf.random.normal(tf.shape(image), stddev=1.5, seed=SEED + 13)
    image = tf.clip_by_value(image + noise, 0.0, 255.0)
    return image


def preprocess_tensor(image: tf.Tensor, model_key: str) -> tf.Tensor:
    if model_key == "mobilenetv2_model":
        return mobilenet_preprocess(image)
    if model_key == "resnet50_model":
        return resnet_preprocess(image)
    if model_key == "efficientnetb0_model":
        # EfficientNetB0 includes rescaling internally and expects [0,255].
        return efficientnet_preprocess(image)
    if model_key == "custom_cnn_model":
        return image / 255.0
    raise ValueError(model_key)


def make_dataset(
    records: Sequence[Dict[str, object]],
    model_key: str,
    batch_size: int,
    training: bool,
    sample_weights: np.ndarray | None = None,
) -> tf.data.Dataset:
    paths = np.asarray([str(r["path"]) for r in records], dtype=str)
    labels = np.asarray([int(r["label"]) for r in records], dtype=np.int32)

    if sample_weights is None:
        sample_weights = np.ones(len(records), dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, sample_weights))

    if training:
        ds = ds.shuffle(
            buffer_size=max(len(records), 1),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def _map(path, label, weight):
        image = decode_resize(path)
        if training:
            image = augment_train(image)
        image = preprocess_tensor(image, model_key)
        y = tf.one_hot(label, depth=NUM_CLASSES)
        return image, y, weight

    options = tf.data.Options()
    options.experimental_deterministic = not training
    ds = ds.with_options(options)
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------
def classifier_head(x: tf.Tensor, dense_units: int = 256) -> tf.Tensor:
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(
        dense_units,
        activation="swish",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(0.25)(x)
    return layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)


def build_transfer_model(model_key: str):
    if model_key == "mobilenetv2_model":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    elif model_key == "resnet50_model":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    elif model_key == "efficientnetb0_model":
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    else:
        raise ValueError(model_key)

    base.trainable = False
    inputs = keras.Input(shape=INPUT_SHAPE, name="image")
    # training=False keeps BatchNorm in inference mode during fine-tuning.
    x = base(inputs, training=False)
    outputs = classifier_head(x)
    model = keras.Model(inputs, outputs, name=model_key)
    return model, base


def _custom_residual_block(
    x: tf.Tensor,
    filters: int,
    stride: int = 1,
    dropout_rate: float = 0.0,
) -> tf.Tensor:
    """Residual block used only by the custom CNN."""
    shortcut = x

    x = layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(5e-5),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(5e-5),
    )(x)
    x = layers.BatchNormalization()(x)

    # Match the shortcut shape whenever spatial size or channel count changes.
    if stride != 1 or int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("swish")(x)
    if dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate)(x)
    return x


def build_custom_cnn_model() -> keras.Model:
    """
    Stronger custom CNN trained from scratch.

    The previous scratch CNN was too weak for the visual variation in the
    reactor dataset. This version uses residual connections, progressively
    deeper feature maps, BatchNorm, spatial dropout, global average pooling,
    and weight regularization. It remains fully independent from the three
    pretrained models.
    """
    inputs = keras.Input(shape=INPUT_SHAPE, name="image")

    # Stem: learn low-level edges, bubbles, foam texture, and reactor boundaries.
    x = layers.Conv2D(
        32,
        5,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(5e-5),
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    # Residual stages. Downsampling happens at the first block of each new stage.
    x = _custom_residual_block(x, 32, stride=1, dropout_rate=0.05)
    x = _custom_residual_block(x, 32, stride=1, dropout_rate=0.05)

    x = _custom_residual_block(x, 64, stride=2, dropout_rate=0.08)
    x = _custom_residual_block(x, 64, stride=1, dropout_rate=0.08)

    x = _custom_residual_block(x, 128, stride=2, dropout_rate=0.10)
    x = _custom_residual_block(x, 128, stride=1, dropout_rate=0.10)

    x = _custom_residual_block(x, 256, stride=2, dropout_rate=0.12)
    x = _custom_residual_block(x, 256, stride=1, dropout_rate=0.12)

    # Classification head. GlobalAveragePooling is much less prone to memorizing
    # exact pixel locations than Flatten(), which is important for new reactors
    # and different camera angles.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        256,
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs, name="custom_cnn_model")


def make_optimizer(learning_rate: float):
    # AdamW is available in modern tf.keras and helps regularize fine-tuning.
    try:
        return keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=WEIGHT_DECAY,
            clipnorm=1.0,
        )
    except Exception:
        return keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=make_optimizer(learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )


def compile_custom_cnn(model: keras.Model) -> None:
    """Compile settings used only for the custom CNN."""
    try:
        optimizer = keras.optimizers.AdamW(
            learning_rate=CUSTOM_CNN_LR,
            weight_decay=CUSTOM_CNN_WEIGHT_DECAY,
            clipnorm=1.0,
        )
    except Exception:
        optimizer = keras.optimizers.Adam(
            learning_rate=CUSTOM_CNN_LR,
            clipnorm=1.0,
        )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=CUSTOM_CNN_LABEL_SMOOTHING
        ),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )


def custom_cnn_callbacks():
    """Longer patience because the scratch CNN learns much more slowly."""
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "custom_cnn_model_full_best.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=CUSTOM_CNN_EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.35,
            patience=CUSTOM_CNN_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(REPORTS_DIR / "custom_cnn_model_full_history.csv")
        ),
        keras.callbacks.TerminateOnNaN(),
    ]


def callbacks_for(model_key: str, phase: str):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{model_key}_{phase}_best.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.35,
            patience=LR_PATIENCE,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(REPORTS_DIR / f"{model_key}_{phase}_history.csv")
        ),
        keras.callbacks.TerminateOnNaN(),
    ]


def enable_fine_tuning(model_key: str, base_model: keras.Model) -> None:
    base_model.trainable = True
    fraction = FINE_TUNE_FRACTION[model_key]
    cutoff = int(round(len(base_model.layers) * (1.0 - fraction)))

    for i, layer in enumerate(base_model.layers):
        if i < cutoff or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_layers = sum(1 for layer in base_model.layers if layer.trainable)
    print(
        f"{model_key}: fine-tuning {trainable_layers}/{len(base_model.layers)} "
        "backbone layers; BatchNorm remains frozen."
    )


# -----------------------------------------------------------------------------
# Evaluation / reporting
# -----------------------------------------------------------------------------
def predict_dataset(model: keras.Model, ds: tf.data.Dataset) -> np.ndarray:
    return np.asarray(model.predict(ds, verbose=1))


def true_labels(records: Sequence[Dict[str, object]]) -> np.ndarray:
    return np.asarray([int(r["label"]) for r in records], dtype=np.int32)


def evaluate_probs(
    model_key: str,
    split_name: str,
    probs: np.ndarray,
    records: Sequence[Dict[str, object]],
    save_details: bool = True,
) -> Dict[str, float]:
    y_true = true_labels(records)
    y_pred = np.argmax(probs, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    print(f"\n{model_key} {split_name}: accuracy={acc:.4f}, macro_f1={macro_f1:.4f}")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    if save_details:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm)
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{model_key} {split_name} confusion matrix")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{model_key}_{split_name}_confusion_matrix.png")
        plt.close(fig)

        detail_rows = []
        for i, r in enumerate(records):
            order = np.argsort(probs[i])[::-1]
            detail_rows.append(
                {
                    "path": r["path"],
                    "subtype": r["subtype"],
                    "group": r["group"],
                    "true_class": r["class_name"],
                    "predicted_class": CLASS_NAMES[int(order[0])],
                    "confidence": float(probs[i][order[0]]),
                    "second_choice": CLASS_NAMES[int(order[1])],
                    "second_confidence": float(probs[i][order[1]]),
                    "correct": int(order[0]) == int(r["label"]),
                }
            )
        with (REPORTS_DIR / f"{model_key}_{split_name}_predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)

        # Medium/Heavy subtype diagnostic: this is the report that tells you if,
        # for example, yellow-heavy is failing while white-heavy works.
        subtype_rows = []
        for class_name in sorted(SUBTYPE_WEIGHT_CLASSES):
            class_idx = CLASS_TO_INDEX[class_name]
            subtypes = sorted(
                {str(r["subtype"]) for r in records if r["class_name"] == class_name}
            )
            for subtype in subtypes:
                ids = [
                    i
                    for i, r in enumerate(records)
                    if r["class_name"] == class_name and r["subtype"] == subtype
                ]
                if not ids:
                    continue
                sub_acc = float(np.mean(y_pred[ids] == class_idx))
                subtype_rows.append(
                    {
                        "class_name": class_name,
                        "subtype": subtype,
                        "count": len(ids),
                        "accuracy": sub_acc,
                    }
                )
        if subtype_rows:
            with (REPORTS_DIR / f"{model_key}_{split_name}_subtype_accuracy.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(
                    f, fieldnames=["class_name", "subtype", "count", "accuracy"]
                )
                writer.writeheader()
                writer.writerows(subtype_rows)

    return {"accuracy": acc, "macro_f1": macro_f1}


def save_training_history(histories: Sequence[keras.callbacks.History], model_key: str) -> None:
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history.get("accuracy", [])
        val_acc += h.history.get("val_accuracy", [])
        loss += h.history.get("loss", [])
        val_loss += h.history.get("val_loss", [])

    if acc:
        plt.figure(figsize=(9, 5))
        plt.plot(acc, label="train")
        plt.plot(val_acc, label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(model_key)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{model_key}_accuracy_curve.png")
        plt.close()
    if loss:
        plt.figure(figsize=(9, 5))
        plt.plot(loss, label="train")
        plt.plot(val_loss, label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(model_key)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{model_key}_loss_curve.png")
        plt.close()


def save_model(model: keras.Model, model_key: str) -> None:
    # Both names contain the SAME selected best-overall model. Inference prefers
    # *_best.keras but also supports the plain .keras name.
    model.save(MODELS_DIR / f"{model_key}_best.keras")
    model.save(MODELS_DIR / f"{model_key}.keras")
    model.save_weights(MODELS_DIR / f"{model_key}.weights.h5")


def clear_memory() -> None:
    keras.backend.clear_session()
    gc.collect()


# -----------------------------------------------------------------------------
# Train one model
# -----------------------------------------------------------------------------
def train_one_model(
    model_key: str,
    train_records: Sequence[Dict[str, object]],
    val_records: Sequence[Dict[str, object]],
    test_records: Sequence[Dict[str, object]],
    train_weights: np.ndarray,
):
    print("\n" + "=" * 80)
    print("TRAINING", model_key)
    print("=" * 80)
    clear_memory()

    batch_size = BATCH_SIZE_CUSTOM if model_key == "custom_cnn_model" else BATCH_SIZE_TRANSFER
    train_ds = make_dataset(
        train_records,
        model_key,
        batch_size,
        training=True,
        sample_weights=train_weights,
    )
    val_ds = make_dataset(val_records, model_key, batch_size, training=False)
    test_ds = make_dataset(test_records, model_key, batch_size, training=False)

    histories = []

    if model_key == "custom_cnn_model":
        model = build_custom_cnn_model()
        compile_custom_cnn(model)
        print(
            f"Custom CNN: training from scratch for up to {CUSTOM_CNN_EPOCHS} epochs "
            f"at lr={CUSTOM_CNN_LR}. Early stopping is enabled."
        )
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=CUSTOM_CNN_EPOCHS,
            callbacks=custom_cnn_callbacks(),
            verbose=1,
        )
        histories.append(history)
    else:
        model, base_model = build_transfer_model(model_key)

        print("Phase 1: train classification head")
        compile_model(model, HEAD_LR)
        h1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=HEAD_EPOCHS,
            callbacks=callbacks_for(model_key, "head"),
            verbose=1,
        )
        histories.append(h1)
        head_best_loss = float(min(h1.history.get("val_loss", [float("inf")])))

        print("Phase 2: fine-tune upper backbone")
        enable_fine_tuning(model_key, base_model)
        compile_model(model, FINETUNE_LR)
        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=FINETUNE_EPOCHS,
            callbacks=callbacks_for(model_key, "finetune"),
            verbose=1,
        )
        histories.append(h2)
        finetune_best_loss = float(min(h2.history.get("val_loss", [float("inf")])))

        # Fine-tuning is helpful only if validation performance actually improves.
        # If it degrades, restore the best frozen-backbone checkpoint instead of
        # forcing the worse fine-tuned model into production.
        if head_best_loss + 1e-8 < finetune_best_loss:
            print(
                f"Fine-tuning did not beat head-only val_loss "
                f"({finetune_best_loss:.5f} vs {head_best_loss:.5f}); "
                "restoring the head-only checkpoint."
            )
            model = keras.models.load_model(
                MODELS_DIR / f"{model_key}_head_best.keras", compile=False
            )

    save_training_history(histories, model_key)
    save_model(model, model_key)

    val_probs = predict_dataset(model, val_ds)
    test_probs = predict_dataset(model, test_ds)
    val_metrics = evaluate_probs(model_key, "validation", val_probs, val_records)
    test_metrics = evaluate_probs(model_key, "test", test_probs, test_records)

    result = {
        "model_key": model_key,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_probs": val_probs,
        "test_probs": test_probs,
    }

    del model, train_ds, val_ds, test_ds
    if model_key != "custom_cnn_model":
        del base_model
    clear_memory()
    return result


# -----------------------------------------------------------------------------
# Probability calibration
# -----------------------------------------------------------------------------
def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    Temperature-scale probabilities without changing a model's class ranking.
    This improves confidence calibration and ensemble reliability.
    """
    temperature = max(float(temperature), 1e-3)
    clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-8, 1.0)
    logits = np.log(clipped) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-12)


def fit_temperature(
    probs: np.ndarray,
    records: Sequence[Dict[str, object]],
) -> float:
    """Fit confidence temperature using validation negative log-likelihood."""
    y_true = true_labels(records)

    def nll(p):
        chosen = np.clip(p[np.arange(len(y_true)), y_true], 1e-8, 1.0)
        return float(-np.mean(np.log(chosen)))

    best_t = 1.0
    best_loss = nll(np.asarray(probs, dtype=np.float64))

    for t in np.linspace(0.40, 4.00, 145):
        scaled = apply_temperature(probs, float(t))
        loss = nll(scaled)
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)

    return best_t


# -----------------------------------------------------------------------------
# Ensemble optimization
# -----------------------------------------------------------------------------
def weighted_probs(model_probs: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    stacked = np.stack(model_probs, axis=0)
    return np.tensordot(weights, stacked, axes=(0, 0))


def optimize_ensemble(
    model_keys: List[str],
    val_probs: List[np.ndarray],
    val_records: Sequence[Dict[str, object]],
) -> np.ndarray:
    y_val = true_labels(val_records)
    rng = np.random.default_rng(SEED)

    candidates = [
        np.ones(len(model_keys), dtype=np.float64) / len(model_keys),
    ]
    # Include every single model as a possible "ensemble" so a weak model can
    # never be forced to hurt the final result.
    for i in range(len(model_keys)):
        w = np.zeros(len(model_keys), dtype=np.float64)
        w[i] = 1.0
        candidates.append(w)

    # Validation-tuned mixtures.
    candidates.extend(rng.dirichlet(np.ones(len(model_keys)), size=5000))

    best_score = -1.0
    best_acc = -1.0
    best_weights = candidates[0]

    for w in candidates:
        probs = weighted_probs(val_probs, np.asarray(w))
        pred = np.argmax(probs, axis=1)
        score = f1_score(y_val, pred, average="macro", zero_division=0)
        acc = accuracy_score(y_val, pred)
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and acc > best_acc
        ):
            best_score = float(score)
            best_acc = float(acc)
            best_weights = np.asarray(w, dtype=np.float64)

    best_weights = best_weights / best_weights.sum()
    print("\nBest validation ensemble:")
    print("  macro_f1:", round(best_score, 4), "accuracy:", round(best_acc, 4))
    for key, w in zip(model_keys, best_weights):
        print(f"  {key}: {w:.4f}")
    return best_weights


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    records = discover_original_records()
    if not records:
        raise RuntimeError(
            "No original images found. Expected class folders in the current directory."
        )

    missing_classes = [
        c for c in CLASS_NAMES if not any(r["class_name"] == c for r in records)
    ]
    if missing_classes:
        raise RuntimeError(f"Missing image data for classes: {missing_classes}")

    print_dataset_summary(records)
    train_records, val_records, test_records = grouped_split(records)
    save_split_manifest(train_records, val_records, test_records)

    print("\nSPLIT SUMMARY")
    print_split_summary("Train", train_records)
    print_split_summary("Validation", val_records)
    print_split_summary("Test", test_records)

    train_weights = calculate_train_weights(train_records)
    print(
        "Train sample weights: min=%.3f mean=%.3f max=%.3f"
        % (train_weights.min(), train_weights.mean(), train_weights.max())
    )

    model_keys = [
        "custom_cnn_model",
        "mobilenetv2_model",
        "resnet50_model",
        "efficientnetb0_model",
    ]

    results = []
    for model_key in model_keys:
        results.append(
            train_one_model(
                model_key,
                train_records,
                val_records,
                test_records,
                train_weights,
            )
        )

    # Calibrate each model's confidence on validation data before combining
    # models. Argmax labels remain unchanged; probability sharpness becomes
    # more trustworthy for ensemble blending and uncertainty detection.
    temperatures = {}
    calibrated_val_probs = []
    calibrated_test_probs = []

    for r in results:
        key = r["model_key"]
        temperature = fit_temperature(r["val_probs"], val_records)
        temperatures[key] = float(temperature)
        calibrated_val_probs.append(
            apply_temperature(r["val_probs"], temperature)
        )
        calibrated_test_probs.append(
            apply_temperature(r["test_probs"], temperature)
        )
        print(f"{key}: validation temperature={temperature:.3f}")

    ensemble_weights = optimize_ensemble(
        model_keys,
        calibrated_val_probs,
        val_records,
    )

    ensemble_val = weighted_probs(calibrated_val_probs, ensemble_weights)
    ensemble_test = weighted_probs(calibrated_test_probs, ensemble_weights)
    ensemble_val_metrics = evaluate_probs(
        "ensemble", "validation", ensemble_val, val_records
    )
    ensemble_test_metrics = evaluate_probs(
        "ensemble", "test", ensemble_test, test_records
    )

    metrics_json = {
        r["model_key"]: {
            "validation": r["val_metrics"],
            "test": r["test_metrics"],
        }
        for r in results
    }
    metrics_json["ensemble"] = {
        "validation": ensemble_val_metrics,
        "test": ensemble_test_metrics,
    }
    with (MODELS_DIR / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)

    ensemble_config = {
        "class_names": CLASS_NAMES,
        "image_size": [IMG_HEIGHT, IMG_WIDTH],
        "weights": {
            key: float(w) for key, w in zip(model_keys, ensemble_weights)
        },
        "temperatures": temperatures,
        "smoothing_window": INFERENCE_SMOOTHING_WINDOW,
        "min_confidence": INFERENCE_MIN_CONFIDENCE,
        "min_margin": INFERENCE_MIN_MARGIN,
        "resize_mode": "resize_with_pad",
        "notes": (
            "Weights optimized on validation macro-F1. No class-specific bias "
            "multipliers are used."
        ),
    }
    with (MODELS_DIR / "ensemble_config.json").open("w", encoding="utf-8") as f:
        json.dump(ensemble_config, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    for r in results:
        print(
            f"{r['model_key']}: "
            f"val macro-F1={r['val_metrics']['macro_f1']:.4f}, "
            f"test macro-F1={r['test_metrics']['macro_f1']:.4f}, "
            f"test accuracy={r['test_metrics']['accuracy']:.4f}"
        )
    print(
        "ensemble: "
        f"val macro-F1={ensemble_val_metrics['macro_f1']:.4f}, "
        f"test macro-F1={ensemble_test_metrics['macro_f1']:.4f}, "
        f"test accuracy={ensemble_test_metrics['accuracy']:.4f}"
    )
    print("\nSaved models -> ./models")
    print("Saved diagnostics -> ./training_reports and ./figures")
    print("Inference weights -> ./models/ensemble_config.json")


if __name__ == "__main__":
    main()