#mobilenet.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Conv2D, MaxPooling2D, Flatten
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
FRAME_INTERVAL = 5


data_dir = '.'


# Explicitly define the classes based on your directory structure
class_names = ['Foam-Heavy', 'Foam-mild', 'Post-Antifoam Addition', 'Foam-Medium', 'No Foam']
num_classes = len(class_names)


print(f"Classes: {class_names}")
print(f"Number of classes: {num_classes}")


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
           img = keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
           img_array = keras.preprocessing.image.img_to_array(img)
           class_images.append(img_array)
          
   # Load images from augmented directory
   augmented_dir = os.path.join(class_dir, 'augmented')
   if os.path.exists(augmented_dir):
       for img_file in os.listdir(augmented_dir):
           if img_file.endswith(('.jpg', '.jpeg', '.png')):
               img_path = os.path.join(augmented_dir, img_file)
               img = keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
               img_array = keras.preprocessing.image.img_to_array(img)
               class_images.append(img_array)
  
   all_images.extend(class_images)
   all_labels.extend([i] * len(class_images))
   print(f"Class {class_name}: {len(class_images)} images (including augmented)")


all_images = np.array(all_images)
all_labels = np.array(all_labels)


print(f"Total images after augmentation: {len(all_images)}")
print(f"Images per class after augmentation: {Counter(all_labels)}")


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


def create_pretrained_model(base_model, model_name, input_shape):
   base = base_model(weights='imagenet', include_top=False, input_shape=input_shape)
   base.trainable = False  # Freeze the base model
  
   model = Sequential([
       Input(shape=input_shape),
       base,
       GlobalAveragePooling2D(),
       Dense(256, activation='relu'),
       Dropout(0.5),
       Dense(num_classes, activation='softmax')
   ])
  
   return model, model_name


input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
models_to_test = [
   create_pretrained_model(MobileNetV2, 'MobileNetV2', input_shape)
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
       train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(BATCH_SIZE, drop_remainder=False)
       val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE, drop_remainder=False)
       history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
       for epoch in range(EPOCHS):
           train_loss = tf.keras.metrics.Mean(name='train_loss')
           train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
           val_loss = tf.keras.metrics.Mean(name='val_loss')
           val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
           for images, labels in train_dataset:
               with tf.GradientTape() as tape:
                   predictions = model(images, training=True)
                   loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
               gradients = tape.gradient(loss, model.trainable_variables)
               model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
          
               train_loss.update_state(loss)
               train_accuracy.update_state(labels, predictions)


           for val_images, val_labels in val_dataset:
               val_predictions = model(val_images, training=False)
               v_loss = tf.keras.losses.categorical_crossentropy(val_labels, val_predictions)
              
               val_loss.update_state(v_loss)
               val_accuracy.update_state(val_labels, val_predictions)


           print(f'Epoch {epoch + 1}, '
                 f'Loss: {train_loss.result():.4f}, '
                 f'Accuracy: {train_accuracy.result():.4f}, '
                 f'Val Loss: {val_loss.result():.4f}, '
                 f'Val Accuracy: {val_accuracy.result():.4f}')


           history['accuracy'].append(train_accuracy.result().numpy())
           history['val_accuracy'].append(val_accuracy.result().numpy())
           history['loss'].append(train_loss.result().numpy())
           history['val_loss'].append(val_loss.result().numpy())
      
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
  
       model.save(f'models/{model_name.lower().replace(" ", "_")}_postdataaug_model.h5')
   except Exception as e:
       print(f"Error during prediction and reporting for {model_name}: {str(e)}")


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
   plt.savefig(f'figures/{model_name.lower().replace(" ", "_")}_postdataaug__confusion_matrix.png')
   plt.close()


print("\nTraining and evaluation complete. Model files and visualizations have been saved.")

