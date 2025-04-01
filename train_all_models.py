import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Basic parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 20

# Directory containing class folders
data_dir = '.'

# Class definitions
class_names = ['Foam-Heavy', 'Foam-mild', 'Post-Antifoam Addition', 'Foam-Medium', 'No Foam']
num_classes = len(class_names)

print(f"Classes: {class_names}")
print(f"Number of classes: {num_classes}")

def load_data():
    """Load images from all class directories"""
    all_images = []
    all_labels = []

    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            continue
            
        class_images = []
        # Load images from main directory
        for img_file in os.listdir(class_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_file)
                try:
                    img = keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                    img_array = keras.preprocessing.image.img_to_array(img)
                    class_images.append(img_array)
                except Exception as e:
                    print(f"Error loading {img_path}: {str(e)}")
                    continue
                
        # Load images from augmented directory
        augmented_dir = os.path.join(class_dir, 'augmented')
        if os.path.exists(augmented_dir):
            for img_file in os.listdir(augmented_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(augmented_dir, img_file)
                    try:
                        img = keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                        img_array = keras.preprocessing.image.img_to_array(img)
                        class_images.append(img_array)
                    except Exception as e:
                        print(f"Error loading {img_path}: {str(e)}")
                        continue
        
        all_images.extend(class_images)
        all_labels.extend([i] * len(class_images))
        print(f"Class {class_name}: {len(class_images)} images (including augmented)")

    return np.array(all_images), np.array(all_labels)

def create_model(model_type, input_shape, num_classes):
    """Create a model of the specified type"""
    if model_type == "mobilenetv2":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_type == "vgg16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_type == "resnet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    base_model.trainable = False  # Freeze base model weights
    
    model = Sequential([
        Input(shape=input_shape),
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_and_save_model(model_type, X_train, y_train, X_test, y_test):
    """Train and save a model of the specified type"""
    print(f"\nTraining {model_type.upper()} model...")
    
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    model = create_model(model_type, input_shape, num_classes)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Create checkpoint callback to save best model
    # Fix: Use .keras extension for TensorFlow 2.17+
    os.makedirs('models', exist_ok=True)
    checkpoint_path = f"models/{model_type}_best.keras"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"{model_type.upper()} - Test accuracy: {test_accuracy:.4f}")
    
    # Generate predictions for confusion matrix and classification report
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Create classification report and confusion matrix
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nClassification Report:")
    print(report)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_type.upper()} Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{model_type}_confusion_matrix.png')
    plt.close()
    
    # Save final model with correct extension for TF 2.17+
    final_model_path = f"models/{model_type}_model.keras"
    
    # Also save with .h5 extension for backward compatibility with your inference code
    h5_model_path = f"models/{model_type}_model.h5"
    
    # Save using both methods for maximum compatibility
    model.save(final_model_path)
    
    # For .h5 compatibility, use SaveModel format with h5 extension
    try:
        model.save(h5_model_path, save_format='h5')
        print(f"Model saved to {h5_model_path} (h5 format)")
    except Exception as e:
        print(f"Warning: Couldn't save in h5 format: {str(e)}")
        # Fallback: try older method for TF compatibility
        try:
            import h5py
            model.save_weights(h5_model_path)
            print(f"Saved model weights to {h5_model_path}")
        except Exception as e2:
            print(f"Warning: Also failed to save weights: {str(e2)}")
    
    print(f"Model saved to {final_model_path} (keras format)")
    
    return {
        'model': model,
        'history': history.history,
        'accuracy': test_accuracy,
        'report': report,
        'cm': cm
    }

def main():
    print("Loading and preprocessing data...")
    X, y = load_data()
    
    print(f"Total images: {len(X)}")
    print(f"Images per class: {Counter(y)}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Convert labels to categorical (one-hot encoding)
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)
    
    # Train all three models
    models_to_train = ['mobilenetv2', 'vgg16', 'resnet50']
    results = {}
    
    for model_type in models_to_train:
        try:
            results[model_type] = train_and_save_model(
                model_type, X_train, y_train_cat, X_test, y_test_cat
            )
        except Exception as e:
            print(f"Error training {model_type}: {str(e)}")
    
    # Print summary of all models
    print("\n===== TRAINING RESULTS =====")
    for model_type, result in results.items():
        print(f"\n{model_type.upper()}: Test Accuracy = {result['accuracy']:.4f}")
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    legend_entries = []
    
    for model_type, result in results.items():
        if 'history' in result and 'accuracy' in result['history']:
            train_line, = plt.plot(result['history']['accuracy'], 
                        linestyle='-', 
                        marker='o', 
                        markersize=4)
            val_line, = plt.plot(result['history']['val_accuracy'], 
                      linestyle='--', 
                      marker='s', 
                      markersize=4)
            
            # Create manual legend entries
            legend_entries.append((train_line, f'{model_type} (Train)'))
            legend_entries.append((val_line, f'{model_type} (Val)'))
    
    # Create legend from the lines and labels we saved
    if legend_entries:
        plt.legend([line for line, label in legend_entries],
                  [label for line, label in legend_entries])
    
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('figures/model_accuracy_comparison.png')
    plt.close()
    
    print("\nTraining complete! Models have been saved to the 'models' directory.")

if __name__ == "__main__":
    # Display TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    main()
