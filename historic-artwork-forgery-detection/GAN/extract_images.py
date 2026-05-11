"""
Randomly samples a specified number of images from a source folder and copies them to an output folder.
"""

import os
import random
import shutil

def sample_images(source_folder, output_folder, num_images=4000):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    for img_file in selected_images:
        shutil.copy2(os.path.join(source_folder, img_file), os.path.join(output_folder, img_file))
    print(f"Copied {len(selected_images)} images.")

if __name__ == "__main__":
    sample_images("images/sample", "data/GAN-data_processed", 4001)
