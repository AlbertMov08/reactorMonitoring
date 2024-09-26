import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Keep this seeding consistent
np.random.seed(42)
tf.random.set_seed(42)

#I set this arbitrarily
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 20

#TODO: Github may not be the best place for storage, eventually modify
data_dir = '.'
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

#dynamic class generation
num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")

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
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(validation_generator)
    print(f"{model_name} - Test accuracy: {test_accuracy:.4f}")
    
    # Generate predictions
    predictions = model.predict(validation_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = validation_generator.classes
    
    # Generate classification report and confusion matrix
    class_names = list(train_generator.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)
    
    results[model_name] = {
        'accuracy': test_accuracy,
        'history': history.history,
        'report': report,
        'cm': cm
    }
    
    # Save the model
    model.save(f'{model_name.lower().replace(" ", "_")}_model.h5')

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
    sns.heatmap(result['cm'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()

print("\nTraining and evaluation complete. Model files and visualizations have been saved.")