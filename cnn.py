import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from collections import Counter

# Keep this seeding consistent
np.random.seed(42)
tf.random.set_seed(42)

#I set this arbitrarily
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 20
FRAME_INTERVAL = 30  # Extract a frame every 30 frames

data_dir = '.'

# Explicitly define the classes based on your directory structure
class_names = ['Foam-Heavy', 'Foam-mild', 'Post-Antifoam Addition', 'Foam-Medium', 'No Foam']
num_classes = len(class_names)

print(f"Classes: {class_names}")
print(f"Number of classes: {num_classes}")

def extract_frames(video_path, output_dir, interval=FRAME_INTERVAL):
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return 0
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error opening video file: {video_path}")
        return 0
    
    count = 0
    frame_count = 0
    
    while True:
        success, frame = video.read()
        if not success:
            break
        if count % interval == 0:
            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            frame_count += 1
        count += 1
    
    video.release()
    return frame_count

# Process videos and extract frames
total_frames = 0
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    if not os.path.exists(class_dir):
        print(f"Directory not found: {class_dir}")
        continue
    for file in os.listdir(class_dir):
        print(file)
        if file.endswith('.MOV'):
            video_path = os.path.join(class_dir, file)
            frames = extract_frames(video_path, class_dir)
            total_frames += frames
            print(f"Extracted {frames} frames from {file}")

print(f"Total frames extracted: {total_frames}")

# Load all images (including extracted frames)
all_images = []
all_labels = []
class_counts = Counter()

for i, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    if not os.path.exists(class_dir):
        print(f"Directory not found: {class_dir}")
        continue
    class_images = []
    for img_file in os.listdir(class_dir):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(class_dir, img_file)
            img = keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = keras.preprocessing.image.img_to_array(img)
            class_images.append(img_array)
    class_counts[i] = len(class_images)
    all_images.extend(class_images)
    all_labels.extend([i] * len(class_images))
    print(f"Loaded {len(class_images)} images for class {class_name}")

print("Class counts before balancing:", class_counts)

# Find the minimum count across all classes
min_count = min(class_counts.values())

# Balance the dataset
balanced_images = []
balanced_labels = []
for i in range(num_classes):
    class_images = [img for img, label in zip(all_images, all_labels) if label == i]
    np.random.shuffle(class_images)
    balanced_images.extend(class_images[:min_count])
    balanced_labels.extend([i] * min_count)

all_images = np.array(balanced_images)
all_labels = np.array(balanced_labels)

print(f"Total images after balancing: {len(all_images)}")
print(f"Images per class after balancing: {Counter(all_labels)}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("Shape of training data:", X_train.shape)
print("Shape of training labels:", y_train.shape)
print("Shape of test data:", X_test.shape)
print("Shape of test labels:", y_test.shape)

# Normalize the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

def create_custom_cnn():
    model = Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_pretrained_model(base_model, model_name):
    base = base_model(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base.trainable = False  # Freeze the base model
    
    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model, model_name

models_to_test = [
    create_custom_cnn(),
    create_pretrained_model(VGG16, 'VGG16'),
    create_pretrained_model(ResNet50, 'ResNet50'),
    create_pretrained_model(MobileNetV2, 'MobileNetV2')
]

results = {}

for model in models_to_test:
    if isinstance(model, tuple):
        model, model_name = model
    else:
        model_name = 'Custom CNN'
    
    print(f"\nTraining {model_name}")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    try:
        # Manual training loop
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(BATCH_SIZE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss, predictions

        @tf.function
        def val_step(images, labels):
            predictions = model(images, training=False)
            loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
            return loss, predictions

        history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

        for epoch in range(EPOCHS):
            # Train
            train_losses = []
            train_accuracies = []
            for images, labels in train_dataset:
                loss, predictions = train_step(images, labels)
                train_losses.append(loss)
                train_accuracies.append(
                    tf.keras.metrics.categorical_accuracy(labels, predictions)
                )
            
            # Validate
            val_losses = []
            val_accuracies = []
            for images, labels in val_dataset:
                loss, predictions = val_step(images, labels)
                val_losses.append(loss)
                val_accuracies.append(
                    tf.keras.metrics.categorical_accuracy(labels, predictions)
                )
            
            # Compute epoch-level metrics
            train_loss = tf.reduce_mean(train_losses)
            train_accuracy = tf.reduce_mean(train_accuracies)
            val_loss = tf.reduce_mean(val_losses)
            val_accuracy = tf.reduce_mean(val_accuracies)

            print(f'Epoch {epoch + 1}, '
                  f'Loss: {train_loss:.4f}, '
                  f'Accuracy: {train_accuracy:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.4f}')

            history['accuracy'].append(train_accuracy.numpy())
            history['val_accuracy'].append(val_accuracy.numpy())
            history['loss'].append(train_loss.numpy())
            history['val_loss'].append(val_loss.numpy())
        
    except Exception as e:
        print(f"Error during training {model_name}: {str(e)}")
        continue
    
    # Evaluate the model
    try:
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"{model_name} - Test accuracy: {test_accuracy:.4f}")
    except Exception as e:
        print(f"Error during evaluation of {model_name}: {str(e)}")
        continue
    
    # Generate predictions
    try:
        predictions = model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
    
        # Generate classification report and confusion matrix
        report = classification_report(y_true, y_pred, target_names=class_names)
        cm = confusion_matrix(y_true, y_pred)
    
        results[model_name] = {
            'accuracy': test_accuracy,
            'history': history,
            'report': report,
            'cm': cm
        }
    
        # Save the model
        model.save(f'{model_name.lower().replace(" ", "_")}_model.h5')
    except Exception as e:
        print(f"Error during prediction and reporting for {model_name}: {str(e)}")
# Plot accuracy comparison
plt.figure(figsize=(10, 6))
for model_name, result in results.items():
    plt.plot(result['history']['accuracy'], label=f'{model_name} (Train)')
    plt.plot(result['history']['val_accuracy'], label=f'{model_name} (Validation)')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('model_accuracy_comparison.png')
plt.close()

# Print final results and classification reports
for model_name, result in results.items():
    print(f"\n{model_name}:")
    print(f"Final accuracy: {result['accuracy']:.4f}")
    print("\nClassification Report:")
    print(result['report'])
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(result['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()

print("\nTraining and evaluation complete. Model files and visualizations have been saved.")