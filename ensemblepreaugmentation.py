import os
import numpy as np
import tensorflow as tf
import torch
from torch import nn
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from tensorflow import keras
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class EnsembleClassifier:
    def __init__(self, cnn_model_path, class_names, img_height=224, img_width=224, device='cuda'):
        """
        Initialize ensemble classifier with CNN and ViT
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.img_height = img_height
        self.img_width = img_width
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        
        # Load your existing CNN model
        print("Loading CNN model...")
        self.cnn_model = tf.keras.models.load_model(cnn_model_path)
        
        # Initialize ViT model
        print("Initializing ViT model...")
        self.vit_model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        )
        self.vit_model.to(self.device)
        self.vit_model.eval()
        
        # Initialize feature extractor for ViT
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            'google/vit-base-patch16-224'
        )
        
        # Initialize weights
        self.cnn_weight = 0.5
        self.vit_weight = 0.5
        print("Model initialization complete")

    def load_and_preprocess_data(self, data_dir, batch_size=32):
        """
        Load and preprocess data from both main directory and augmented subdirectory
        """
        print("Loading images from:", data_dir)
        all_images = []
        all_labels = []
        
        for i, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Directory not found: {class_dir}")
                continue
            
            # Load images from main directory and augmented subdirectory
            image_paths = []
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_dir, img_file))
            
            
            for img_path in image_paths:
                try:
                    img = keras.preprocessing.image.load_img(img_path, target_size=(self.img_height, self.img_width))
                    img_array = keras.preprocessing.image.img_to_array(img)
                    all_images.append(img_array)
                    all_labels.append(i)
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")
                    continue
            
            print(f"Loaded {len(image_paths)} images for class {class_name}")
            
        return np.array(all_images), np.array(all_labels)
    
    def predict(self, image):
        """
        Make ensemble prediction on a single image
        """
        # Ensure image is in the right format for CNN
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(image, np.ndarray) and image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # CNN prediction
        try:
            cnn_input = tf.convert_to_tensor(image)
            cnn_input = tf.expand_dims(cnn_input, 0)
            cnn_input = tf.cast(cnn_input, tf.float32) / 255.0
            cnn_probs = self.cnn_model.predict(cnn_input, verbose=0)[0]
        except Exception as e:
            print(f"Error in CNN prediction: {str(e)}")
            return None, None
        
        # ViT prediction
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype('uint8'))
            
            # Prepare image for ViT
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                vit_probs = torch.softmax(outputs.logits, dim=-1)
                vit_probs = vit_probs[0].cpu().numpy()
        except Exception as e:
            print(f"Error in ViT prediction: {str(e)}")
            return None, None
        
        # Combine predictions
        final_probs = (self.cnn_weight * cnn_probs + 
                      self.vit_weight * vit_probs)
        ensemble_prediction = np.argmax(final_probs)
        
        return final_probs, ensemble_prediction

    def evaluate(self, X_test, y_test, batch_size=32):
        """
        Evaluate the ensemble model on test data
        """
        predictions = []
        valid_indices = []
        
        print(f"Evaluating {len(X_test)} test samples...")
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i + batch_size]
            batch_preds = []
            for j, image in enumerate(batch):
                probs, _ = self.predict(image)
                if probs is not None:
                    batch_preds.append(probs)
                    valid_indices.append(i + j)
            predictions.extend(batch_preds)
            if (i + batch_size) % 100 == 0:
                print(f"Processed {min(i + batch_size, len(X_test))} images...")
        
        predictions = np.array(predictions)
        valid_indices = np.array(valid_indices)
        y_test_valid = y_test[valid_indices]
        
        y_pred = np.argmax(predictions, axis=1)
        y_true = y_test_valid if len(y_test_valid.shape) == 1 else np.argmax(y_test_valid, axis=1)
        
        # Generate metrics
        accuracy = np.mean(y_pred == y_true)
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'predictions': predictions
        }

    def calibrate_weights(self, val_images, val_labels, metric='accuracy'):
        """
        Calibrate ensemble weights using validation data
        """
        print("Calibrating ensemble weights...")
        best_metric = float('-inf')
        best_weights = (self.cnn_weight, self.vit_weight)
        
        weight_options = np.linspace(0.1, 0.9, 9)
        for cnn_w in weight_options:
            vit_w = 1 - cnn_w
            self.cnn_weight = cnn_w
            self.vit_weight = vit_w
            
            print(f"Testing weights - CNN: {cnn_w:.1f}, ViT: {vit_w:.1f}")
            results = self.evaluate(val_images, val_labels)
            current_metric = results['accuracy']
            print(f"Current accuracy: {current_metric:.4f}")
            
            if current_metric > best_metric:
                best_metric = current_metric
                best_weights = (cnn_w, vit_w)
        
        self.cnn_weight, self.vit_weight = best_weights
        print(f"Best weights found - CNN: {best_weights[0]:.2f}, ViT: {best_weights[1]:.2f}")
        return best_weights

def plot_results(results, class_names, save_dir='figures'):
    """
    Plot and save evaluation results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        results['confusion_matrix'], 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Ensemble Model PreAugmentation Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(save_dir, 'ensemble_preaugmentation_confusion_matrix.png'))
    plt.close()

def main():
    # rand seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    torch.manual_seed(42)
    
    # Your existing constants
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    class_names = ['Foam-Heavy', 'Foam-mild', 'Post-Antifoam Addition', 'Foam-Medium', 'No Foam']
    data_dir = '.'
    
    # Initialize ensemble
    print("Initializing ensemble classifier...")
    ensemble = EnsembleClassifier(
        cnn_model_path='models/custom_cnn_model.h5',
        class_names=class_names,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH
    )
    
    # Load data
    print("Loading and preprocessing data...")
    X, y = ensemble.load_and_preprocess_data(data_dir)
    print(f"Loaded {len(X)} total images")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Calibrate weights
    best_weights = ensemble.calibrate_weights(X_test, y_test)
    
    # Final evaluation
    print("\nEvaluating final ensemble model...")
    results = ensemble.evaluate(X_test, y_test)
    print("\nEnsemble Model Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['report'])
    
    # Plot results
    plot_results(results, class_names)
    print("\nEvaluation complete. Results have been saved to the figures directory.")

if __name__ == "__main__":
    main()