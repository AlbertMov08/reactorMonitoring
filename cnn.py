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

# Create ImageDataGenerator for data augmentation and preprocessing
def extract_frames(video_path, output_dir, interval=FRAME_INTERVAL):
    video = cv2.VideoCapture(video_path)
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
    for file in os.listdir(class_dir):
        if file.endswith('.mov'):
            video_path = os.path.join(class_dir, file)
            frames = extract_frames(video_path, class_dir)
            total_frames += frames
            print(f"Extracted {frames} frames from {file}")

print(f"Total frames extracted: {total_frames}")

# Prepare data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Load all images (including extracted frames)
all_images = []
all_labels = []

for i, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    for img_file in os.listdir(class_dir):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(class_dir, img_file)
            img = keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = keras.preprocessing.image.img_to_array(img)
            all_images.append(img_array)
            all_labels.append(i)

all_images = np.array(all_images)
all_labels = np.array(all_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Create data generators
train_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
test_generator = ImageDataGenerator(rescale=1./255).flow(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

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
    
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_generator,
        validation_steps=len(X_test) // BATCH_SIZE
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"{model_name} - Test accuracy: {test_accuracy:.4f}")
    
    # Generate predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Generate classification report and confusion matrix
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)
    
    results[model_name] = {
        'accuracy': test_accuracy,
        'history': history.history,
        'report': report,
        'cm': cm
    }
    
    # Save the model
    model.save(f'models/{model_name.lower().replace(" ", "_")}_model.h5')

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
for model_name, result in results.items():
    plt.plot(result['history']['accuracy'], label=f'{model_name} (Train)')
    plt.plot(result['history']['val_accuracy'], label=f'{model_name} (Validation)')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('figures/model_accuracy_comparison.png')
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
    plt.savefig(f'figures/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()

print("\nTraining and evaluation complete. Model files and visualizations have been saved.")