# src/data.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# ---------- Constants ----------

IMG_SIZE   = (150, 150)
BATCH_SIZE = 32
SEED       = 42
AUTOTUNE   = tf.data.AUTOTUNE

# ---------- Collection ----------
def collect_paths_and_labels(data_dir):
    """Walk folder-per-class dir → (paths, labels, class_names)."""
    classes = sorted(os.listdir(data_dir))
    paths, labels = [], []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        
        if not os.path.isdir(cls_dir):
            continue
        
        for fname in os.listdir(cls_dir):
            paths.append(os.path.join(cls_dir, fname))
            labels.append(idx)
    return np.array(paths), np.array(labels), classes

# ---------- Splitting ----------
def stratified_split(paths, labels, ratios=(0.7, 0.15, 0.15), seed=SEED):
    """Stratified train/val/test split. Returns 6 arrays"""
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=ratios[1] + ratios[2],
        stratify=labels, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=ratios[2] / (ratios[1] + ratios[2]),
        stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------- Preprocessing ----------
def load_and_preprocess(path, label):
    """Load image from path, resize, normalize to [0, 1]."""
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


# ---------- Augmentation ----------
def make_augmentation():
    """
    Augmentation layers — applied ONLY to the training set.
    Order: flip → rotation → zoom → contrast
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),          # ±10% of 2π
        tf.keras.layers.RandomZoom(0.1),              # ±10%
        tf.keras.layers.RandomContrast(0.1),          # subtle ±10%
    ], name="augmentation")


# ---------- Pipeline ----------
def make_dataset(paths, labels, training=False):
    """ Make a tf.data.Dataset from paths and labels, with optional augmentation."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    
    if training:
        augment = make_augmentation()
        ds = ds.map(
            lambda x, y: (augment(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )

    ds = ds.prefetch(AUTOTUNE)
    return ds

def get_datasets(data_dir, ratios=(0.7, 0.15, 0.15)):
    """One-call API for notebooks."""
    paths, labels, classes = collect_paths_and_labels(data_dir)
    Xtr, Xva, Xte, ytr, yva, yte = stratified_split(paths, labels, ratios)
    train_ds = make_dataset(Xtr, ytr, training=True)
    val_ds   = make_dataset(Xva, yva)
    test_ds  = make_dataset(Xte, yte)
    return train_ds, val_ds, test_ds, classes