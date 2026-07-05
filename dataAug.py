import os
import numpy as np
import cv2
from PIL import Image
from PIL import ImageEnhance
from numpy.random import randint, uniform


# Keep seeding consistent
np.random.seed(42)


# Configuration
IMG_HEIGHT, IMG_WIDTH = 224, 224
TARGET_SIZE = 200  # Number of images wanted per class


# Explicitly define the classes
class_names = ['Foam-Heavy', 'Foam-mild', 'Post-Antifoam Addition', 'Foam-Medium', 'No Foam']


def augment_image(image):
   """
   Apply various augmentations to the image using PIL
   """
   # to PIL Image if np array
   if isinstance(image, np.ndarray):
       image = Image.fromarray(image)
  
   # rotation (-15 to 15 degrees)
   if randint(0, 2):
       angle = uniform(-15, 15)
       image = image.rotate(angle, expand=False, resample=Image.BILINEAR)
  
   # horizontal flip
   if randint(0, 2):
       image = image.transpose(Image.FLIP_LEFT_RIGHT)
  
   # brightness adjustment (0.8 to 1.2)
   if randint(0, 2):
       enhancer = ImageEnhance.Brightness(image)
       factor = uniform(0.8, 1.2)
       image = enhancer.enhance(factor)
  
   # contrast adjustment (0.8 to 1.2)
   if randint(0, 2):
       enhancer = ImageEnhance.Contrast(image)
       factor = uniform(0.8, 1.2)
       image = enhancer.enhance(factor)
  
   # Convert back to numpy array
   return np.array(image)


def create_augmented_images(class_dir, target_size):
   """
   Create augmented images for a specific class directory
   """
   if not os.path.exists(class_dir):
       print(f"Directory not found: {class_dir}")
       return
  
   # augmented directory if it dne
   augmented_dir = os.path.join(class_dir, 'augmented')
   os.makedirs(augmented_dir, exist_ok=True)
  
   # existing images
   existing_images = [f for f in os.listdir(class_dir)
                     if f.endswith(('.jpg', '.jpeg', '.png'))
                     and os.path.isfile(os.path.join(class_dir, f))
                     and 'augmented' not in f]  # Exclude previously augmented images
  
   print(f"\nProcessing {class_dir}")
   print(f"Found {len(existing_images)} existing images")
  
   # Calculate need
   num_augmented_needed = target_size - len(existing_images)
  
   if num_augmented_needed <= 0:
       print("No augmentation needed - already have enough images")
       return
  
   print(f"Need to generate {num_augmented_needed} augmented images")
  
   aug_counter = 0
  
   while aug_counter < num_augmented_needed:
       # Cycle
       for img_file in existing_images:
           if aug_counter >= num_augmented_needed:
               break
           img_path = os.path.join(class_dir, img_file)
           try:
               img = Image.open(img_path)
               img = img.convert('RGB')  # Ensure RGB format
               img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.LANCZOS)
               aug_img = augment_image(np.array(img))
              
               output_path = os.path.join(augmented_dir,
                                        f'aug_{os.path.splitext(img_file)[0]}_{aug_counter}.jpg')
              
               Image.fromarray(aug_img).save(output_path, quality=95)
              
               aug_counter += 1
               if aug_counter % 10 == 0:
                   print(f"Generated {aug_counter}/{num_augmented_needed} augmented images")
              
           except Exception as e:
               print(f"Error processing {img_file}: {str(e)}")
               continue


def main():
   print("Starting image augmentation process...")
  
   for class_name in class_names:
       class_dir = os.path.join('.', class_name)
       create_augmented_images(class_dir, TARGET_SIZE)
  
   print("\nAugmentation complete!")
  
   for class_name in class_names:
       class_dir = os.path.join('.', class_name)
       if os.path.exists(class_dir):
           original_count = len([f for f in os.listdir(class_dir)
                               if f.endswith(('.jpg', '.jpeg', '.png'))
                               and os.path.isfile(os.path.join(class_dir, f))
                               and 'augmented' not in f])
          
           augmented_dir = os.path.join(class_dir, 'augmented')
           augmented_count = len([f for f in os.listdir(augmented_dir)
                                if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(augmented_dir) else 0
          
           print(f"\n{class_name}:")
           print(f"  Original images: {original_count}")
           print(f"  Augmented images: {augmented_count}")
           print(f"  Total images: {original_count + augmented_count}")


if __name__ == "__main__":
   main()

