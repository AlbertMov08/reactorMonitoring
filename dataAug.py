"""
dataAug.py

Dataset audit + inspectable augmentation generator for the reactor foam project.

Design goals
------------
1. Preserve the ORIGINAL photos exactly as they are.
2. Keep Foam-Medium / Foam-Heavy visual variants in separate subtype folders.
3. Generate one controlled copy for each augmentation type so you can inspect
   exactly what happened to each source image.
4. Write manifests / quality reports so bad files and duplicates are visible.
5. Do NOT inflate a tiny subtype into hundreds of near-identical copies.

IMPORTANT
---------
The companion train_models_fixed.py performs the *real training augmentation*
on-the-fly AFTER the train/validation/test split. That avoids data leakage.
The augmented/ folders produced here are primarily for inspection/auditing and
can optionally be used later, but the safest default training path ignores them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEED = 42
DATA_DIR = Path(".")
IMG_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUGMENTED_DIR_NAME = "augmented"

CLASS_NAMES = [
    "Foam-Heavy",
    "Foam-Mild",
    "Post-Antifoam Addition",
    "Foam-Medium",
    "No Foam",
]

# Rebuilding prevents stale augmented copies from older code from being mixed in.
REBUILD_AUGMENTED = False

# One inspectable copy per source image per transformation.
# These are deliberately realistic, mild camera/lighting changes.
AUGMENTATION_TYPES = [
    "small_rotation",
    "horizontal_flip",
    "brightness",
    "contrast",
    "saturation",
    "zoom_translate",
]

# A warning only. Real variety matters much more than creating many synthetic
# copies of the same original.
MIN_REAL_IMAGES_PER_SUBTYPE_WARNING = 20

# Quality report thresholds are advisory only; images are NOT deleted.
VERY_LOW_RESOLUTION_SIDE = 160
VERY_BLURRY_LAPLACIAN_VARIANCE = 12.0

REPORT_DIR = Path("dataset_reports")
REPORT_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def stable_rng(*parts: str) -> np.random.Generator:
    """Deterministic RNG per image/augmentation so reruns are reproducible."""
    key = "|".join(parts).encode("utf-8", errors="ignore")
    digest = hashlib.sha256(key).digest()
    seed = int.from_bytes(digest[:8], "little") ^ SEED
    return np.random.default_rng(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def subtype_for(class_dir: Path, image_path: Path) -> str:
    """
    First folder under the class is treated as the visual subtype.

    Examples
    --------
    Foam-Heavy/yellow_heavy/img.jpg -> yellow_heavy
    Foam-Heavy/yellow_heavy/video_01/frame001.jpg -> yellow_heavy
    Foam-Heavy/img.jpg -> root
    """
    rel_parent = image_path.parent.relative_to(class_dir)
    if str(rel_parent) == ".":
        return "root"
    return rel_parent.parts[0]


def collect_originals(class_dir: Path) -> List[Path]:
    originals: List[Path] = []
    if not class_dir.exists():
        return originals

    for root, dirs, files in os.walk(class_dir):
        root_path = Path(root)
        rel = root_path.relative_to(class_dir)
        rel_parts = [p.lower() for p in rel.parts]

        # Never recurse through previously generated data.
        if AUGMENTED_DIR_NAME.lower() in rel_parts:
            dirs[:] = []
            continue

        # Prevent os.walk from entering augmented at the next level.
        dirs[:] = [d for d in dirs if d.lower() != AUGMENTED_DIR_NAME.lower()]

        for filename in files:
            p = root_path / filename
            if is_image(p):
                originals.append(p)

    return sorted(originals)


def image_quality(path: Path) -> Dict[str, object]:
    """Return non-destructive quality measurements for one image."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        width, height = im.size
        arr_rgb = np.asarray(im)

    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(gray.mean())

    warnings: List[str] = []
    if min(width, height) < VERY_LOW_RESOLUTION_SIDE:
        warnings.append("very_low_resolution")
    if blur_score < VERY_BLURRY_LAPLACIAN_VARIANCE:
        warnings.append("very_blurry")
    if mean_brightness < 20:
        warnings.append("very_dark")
    elif mean_brightness > 240:
        warnings.append("very_bright")

    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / max(height, 1), 4),
        "blur_score": round(blur_score, 3),
        "mean_brightness": round(mean_brightness, 3),
        "quality_warning": ";".join(warnings),
    }



def dhash64(path: Path) -> int:
    """64-bit difference hash used only to flag visually similar images."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        im = im.resize((9, 8), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def hamming_distance64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def prepare_image(path: Path) -> Image.Image:
    """Correct EXIF orientation, convert RGB, and resize with aspect ratio kept."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        # Match the training/inference geometry: preserve full image and pad.
        contained = ImageOps.contain(im, IMG_SIZE, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", IMG_SIZE, (0, 0, 0))
        x = (IMG_SIZE[0] - contained.width) // 2
        y = (IMG_SIZE[1] - contained.height) // 2
        canvas.paste(contained, (x, y))
        return canvas


def median_fill(image: Image.Image) -> Tuple[int, int, int]:
    arr = np.asarray(image)
    med = np.median(arr.reshape(-1, 3), axis=0)
    return tuple(int(v) for v in med)


def apply_augmentation(
    image: Image.Image,
    augmentation_type: str,
    rng: np.random.Generator,
) -> Tuple[Image.Image, Dict[str, float]]:
    """Apply exactly ONE realistic transformation and return its parameters."""
    params: Dict[str, float] = {}

    if augmentation_type == "small_rotation":
        angle = float(rng.uniform(-8.0, 8.0))
        params["angle_degrees"] = round(angle, 4)
        out = image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=median_fill(image),
        )
        return out, params

    if augmentation_type == "horizontal_flip":
        return ImageOps.mirror(image), params

    if augmentation_type == "brightness":
        factor = float(rng.uniform(0.85, 1.15))
        params["factor"] = round(factor, 4)
        return ImageEnhance.Brightness(image).enhance(factor), params

    if augmentation_type == "contrast":
        factor = float(rng.uniform(0.85, 1.15))
        params["factor"] = round(factor, 4)
        return ImageEnhance.Contrast(image).enhance(factor), params

    if augmentation_type == "saturation":
        factor = float(rng.uniform(0.90, 1.10))
        params["factor"] = round(factor, 4)
        return ImageEnhance.Color(image).enhance(factor), params

    if augmentation_type == "zoom_translate":
        zoom = float(rng.uniform(1.00, 1.08))
        shift_x = float(rng.uniform(-0.035, 0.035))
        shift_y = float(rng.uniform(-0.035, 0.035))
        params.update(
            {
                "zoom": round(zoom, 4),
                "shift_x_fraction": round(shift_x, 4),
                "shift_y_fraction": round(shift_y, 4),
            }
        )

        w, h = image.size
        zw, zh = max(w, int(round(w * zoom))), max(h, int(round(h * zoom)))
        enlarged = image.resize((zw, zh), Image.Resampling.BILINEAR)

        base_left = (zw - w) / 2.0
        base_top = (zh - h) / 2.0
        left = int(round(base_left + shift_x * w))
        top = int(round(base_top + shift_y * h))
        left = max(0, min(left, zw - w))
        top = max(0, min(top, zh - h))
        return enlarged.crop((left, top, left + w, top + h)), params

    raise ValueError(f"Unknown augmentation type: {augmentation_type}")


def output_dir_for(class_dir: Path, aug_type: str, image_path: Path) -> Path:
    """Mirror the original folder hierarchy underneath each augmentation."""
    rel_parent = image_path.parent.relative_to(class_dir)
    if str(rel_parent) == ".":
        rel_parent = Path("root")
    return class_dir / AUGMENTED_DIR_NAME / aug_type / rel_parent


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------
def process_dataset() -> None:
    print("Starting foam dataset audit + augmentation generation")
    print("Original photos will NOT be modified.\n")

    quality_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    duplicate_rows: List[Dict[str, object]] = []
    perceptual_rows: List[Dict[str, object]] = []
    global_hash_owner: Dict[str, Path] = {}

    for class_name in CLASS_NAMES:
        class_dir = DATA_DIR / class_name
        if not class_dir.exists():
            print(f"WARNING: missing class folder: {class_dir}")
            continue

        aug_root = class_dir / AUGMENTED_DIR_NAME
        if REBUILD_AUGMENTED and aug_root.exists():
            print(f"Rebuilding generated augmentations: {aug_root}")
            shutil.rmtree(aug_root)

        originals = collect_originals(class_dir)
        subtype_counts: Dict[str, int] = {}
        class_perceptual_hashes: List[Tuple[Path, int]] = []

        print("=" * 72)
        print(f"{class_name}: {len(originals)} original image(s)")

        for image_path in originals:
            subtype = subtype_for(class_dir, image_path)
            subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1

            try:
                file_hash = sha256_file(image_path)
                q = image_quality(image_path)
                perceptual_hash = dhash64(image_path)
            except Exception as e:
                print(f"  ERROR reading {image_path}: {e}")
                quality_rows.append(
                    {
                        "class_name": class_name,
                        "subtype": subtype,
                        "path": str(image_path),
                        "sha256": "",
                        "width": "",
                        "height": "",
                        "aspect_ratio": "",
                        "blur_score": "",
                        "mean_brightness": "",
                        "quality_warning": f"read_error:{e}",
                        "exact_duplicate_of": "",
                    }
                )
                continue

            duplicate_of = ""
            if file_hash in global_hash_owner:
                duplicate_of = str(global_hash_owner[file_hash])
                duplicate_rows.append(
                    {
                        "duplicate": str(image_path),
                        "original": duplicate_of,
                        "sha256": file_hash,
                    }
                )
            else:
                global_hash_owner[file_hash] = image_path

            if not duplicate_of:
                class_perceptual_hashes.append((image_path, perceptual_hash))

            quality_rows.append(
                {
                    "class_name": class_name,
                    "subtype": subtype,
                    "path": str(image_path),
                    "sha256": file_hash,
                    **q,
                    "exact_duplicate_of": duplicate_of,
                }
            )

            # Do not multiply exact duplicate files again.
            if duplicate_of:
                continue

            try:
                source_image = prepare_image(image_path)
            except Exception as e:
                print(f"  ERROR preparing {image_path}: {e}")
                continue

            for aug_type in AUGMENTATION_TYPES:
                rng = stable_rng(class_name, str(image_path), aug_type)
                try:
                    aug_image, params = apply_augmentation(source_image, aug_type, rng)
                    out_dir = output_dir_for(class_dir, aug_type, image_path)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{image_path.stem}__{aug_type}.jpg"
                    aug_image.save(out_path, format="JPEG", quality=95, optimize=True)

                    manifest_rows.append(
                        {
                            "class_name": class_name,
                            "subtype": subtype,
                            "source_path": str(image_path),
                            "output_path": str(out_path),
                            "augmentation": aug_type,
                            "parameters_json": json.dumps(params, sort_keys=True),
                        }
                    )
                except Exception as e:
                    print(f"  ERROR augmenting {image_path} with {aug_type}: {e}")

        # Flag visually near-duplicate frames. Nothing is deleted.
        for i in range(len(class_perceptual_hashes)):
            path_a, hash_a = class_perceptual_hashes[i]
            for j in range(i + 1, len(class_perceptual_hashes)):
                path_b, hash_b = class_perceptual_hashes[j]
                distance = hamming_distance64(hash_a, hash_b)
                if distance <= 4:
                    perceptual_rows.append(
                        {
                            "class_name": class_name,
                            "image_a": str(path_a),
                            "image_b": str(path_b),
                            "dhash_distance": distance,
                        }
                    )

        if subtype_counts:
            print("  Real-image subtype counts:")
            for subtype, count in sorted(subtype_counts.items()):
                marker = ""
                if class_name in {"Foam-Medium", "Foam-Heavy"} and count < MIN_REAL_IMAGES_PER_SUBTYPE_WARNING:
                    marker = "  <-- NEED MORE REAL VARIETY"
                print(f"    {subtype}: {count}{marker}")

    write_csv(
        REPORT_DIR / "dataset_quality_report.csv",
        quality_rows,
        [
            "class_name",
            "subtype",
            "path",
            "sha256",
            "width",
            "height",
            "aspect_ratio",
            "blur_score",
            "mean_brightness",
            "quality_warning",
            "exact_duplicate_of",
        ],
    )
    write_csv(
        REPORT_DIR / "augmentation_manifest.csv",
        manifest_rows,
        [
            "class_name",
            "subtype",
            "source_path",
            "output_path",
            "augmentation",
            "parameters_json",
        ],
    )
    write_csv(
        REPORT_DIR / "exact_duplicates.csv",
        duplicate_rows,
        ["duplicate", "original", "sha256"],
    )
    write_csv(
        REPORT_DIR / "perceptual_near_duplicates.csv",
        perceptual_rows,
        ["class_name", "image_a", "image_b", "dhash_distance"],
    )

    print("\nDone.")
    print(f"Quality report: {REPORT_DIR / 'dataset_quality_report.csv'}")
    print(f"Augmentation manifest: {REPORT_DIR / 'augmentation_manifest.csv'}")
    print(f"Duplicate report: {REPORT_DIR / 'exact_duplicates.csv'}")
    print(
        f"Near-duplicate report: "
        f"{REPORT_DIR / 'perceptual_near_duplicates.csv'}"
    )
    print("\nGenerated folder layout example:")
    print("  Foam-Heavy/augmented/small_rotation/yellow_heavy/video_01/...")
    print("  Foam-Heavy/augmented/brightness/yellow_heavy/video_01/...")
    print("  Foam-Medium/augmented/zoom_translate/cloudy/video_02/...")
    print("\nNOTE: train_models_fixed.py performs fresh augmentation on-the-fly on TRAIN data only.")


if __name__ == "__main__":
    process_dataset()