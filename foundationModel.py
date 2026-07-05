#foundationModel.py
import os
import numpy as np
import cv2
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from tqdm import tqdm
from zeroShotOpenAI import zeroShotOpenAIModel

# Keep this seeding consistent
np.random.seed(42)

IMG_HEIGHT, IMG_WIDTH = 224, 224
FRAME_INTERVAL = 5

data_dir = '.'

# Explicitly define the classes based on your directory structure
class_names = ['Foam-Heavy', 'Foam-Mild', 'Post-Antifoam Addition', 'Foam-Medium', 'No Foam','Error']
num_classes = len(class_names)

print(f"Classes: {class_names}")
print(f"Number of classes: {num_classes}")

# def extract_frames(video_path, output_dir, interval=FRAME_INTERVAL):
#     if not os.path.exists(video_path):
#         print(f"Video file not found: {video_path}")
#         return 0
    
#     video = cv2.VideoCapture(video_path)
#     if not video.isOpened():
#         print(f"Error opening video file: {video_path}")
#         return 0
    
#     count = 0
#     frame_count = 0
    
#     while True:
#         success, frame = video.read()
#         if not success:
#             break
#         if count % interval == 0:
#             frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
#             output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
#             cv2.imwrite(output_path, frame)
#             frame_count += 1
#         count += 1
    
#     video.release()
#     return frame_count

# Process videos and extract frames
# total_frames = 0
# for class_name in class_names:
#     class_dir = os.path.join(data_dir, class_name)
#     if not os.path.exists(class_dir):
#         print(f"Directory not found: {class_dir}")
#         continue
#     for file in os.listdir(class_dir):
#         if file.endswith('.MOV'):
#             video_path = os.path.join(class_dir, file)
#             frames = extract_frames(video_path, class_dir)
#             total_frames += frames
#             print(f"Extracted {frames} frames from {file}")

# print(f"Total frames extracted: {total_frames}")

all_images = []
all_labels = []

for i, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    if not os.path.exists(class_dir):
        print(f"Directory not found: {class_dir}")
        continue
    for img_file in os.listdir(class_dir):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(class_dir, img_file)
            all_images.append(img_path)
            all_labels.append(i)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Perform classification using OpenAI API
y_pred = []
for image_path in tqdm(X_test, desc="Classifying images"):
    try:
        prediction = zeroShotOpenAIModel(image_path)
        y_pred.append(class_names.index(prediction))
    except:
        y_pred.append(class_names.index('Error'))


# Calculate metrics
accuracy = np.mean(np.array(y_pred) == np.array(y_test))
report = classification_report(y_test, y_pred, target_names=class_names)
cm = confusion_matrix(y_test, y_pred)

# Print results
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('OpenAI API Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('figures/openai_api_confusion_matrix.png')
plt.close()

print("\nClassification complete. Results and visualizations have been saved.")