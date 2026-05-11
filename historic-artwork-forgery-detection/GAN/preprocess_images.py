"""
preprocess_images.py
--------------------
Preprocesses images (resize, normalize, grayscale, etc.) for GAN training.
Adjust input/output paths as needed for your folder structure.
"""

import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import cv2

tfa.options.disable_custom_kernel()

INPUT_DIR = "data/GAN-data_processed"
OUTPUT_DIR = "data/GAN-processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_style_perturbation(image):
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.05)
    return tf.clip_by_value(image + noise, -1.0, 1.0)

def color_space_shift(image):
    hsv = tf.image.rgb_to_hsv((image + 1) / 2)
    hue_shift = tf.random.uniform([], -0.05, 0.05)
    sat_shift = tf.random.uniform([], -0.1, 0.1)
    val_shift = tf.random.uniform([], -0.05, 0.05)
    hsv = tf.stack([
        tf.clip_by_value(hsv[..., 0] + hue_shift, 0, 1),
        tf.clip_by_value(hsv[..., 1] + sat_shift, 0, 1),
        tf.clip_by_value(hsv[..., 2] + val_shift, 0, 1)
    ], axis=-1)
    return tf.image.hsv_to_rgb(hsv) * 2 - 1

def average_color_smudge(image, patch_size=20, num_patches=8):
    img_np = ((image + 1) * 127.5).numpy().astype(np.uint8)
    h, w, _ = img_np.shape
    for _ in range(num_patches):
        x = np.random.randint(0, w - patch_size)
        y = np.random.randint(0, h - patch_size)
        patch = img_np[y:y+patch_size, x:x+patch_size]
        mean_color = patch.mean(axis=(0, 1), keepdims=True)
        img_np[y:y+patch_size, x:x+patch_size] = mean_color.astype(np.uint8)
    img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32) / 127.5 - 1.0
    return img_tensor

def invisible_noise(image):
    adv_noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.02)
    return tf.clip_by_value(image + adv_noise, -1.0, 1.0)

def cutout(image, mask_size=50, alpha=0.3):
    img_np = ((image + 1) * 127.5).numpy().astype(np.uint8)
    h, w, _ = img_np.shape
    x = np.random.randint(0, w - mask_size)
    y = np.random.randint(0, h - mask_size)
    gray_patch = np.full((mask_size, mask_size, 3), 127, dtype=np.uint8)
    blended_patch = (alpha * gray_patch + (1 - alpha) * img_np[y:y+mask_size, x:x+mask_size]).astype(np.uint8)
    img_np[y:y+mask_size, x:x+mask_size] = blended_patch
    img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32) / 127.5 - 1.0
    return img_tensor

def jpeg_compression_artifact(image, quality=75):
    img_np = ((image + 1) * 127.5).numpy().astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img_np, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
    img_tensor = tf.convert_to_tensor(decimg, dtype=tf.float32) / 127.5 - 1.0
    return img_tensor

def preprocess_single_image(image_path):
    from PIL import Image, ImageOps
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img = tf.convert_to_tensor(np.array(img), dtype=tf.float32) / 127.5 - 1.0

    img = tfa.image.gaussian_filter2d(img, filter_shape=(7, 7), sigma=1.0)
    img = tf.image.adjust_brightness(img, delta=0.05)

    img = apply_style_perturbation(img)
    img = color_space_shift(img)
    img = average_color_smudge(img)
    img = invisible_noise(img)
    img = cutout(img)  # Modified version here
    img = jpeg_compression_artifact(img)

    return img

def save_image(tensor, filename):
    img = (tensor + 1.0) * 127.5
    img = tf.cast(img, tf.uint8).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

if __name__ == "__main__":
    image_paths = [os.path.join(INPUT_DIR, fname) for fname in os.listdir(INPUT_DIR)
                   if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for path in image_paths:
        processed_img = preprocess_single_image(path)
        filename = os.path.basename(path)
        save_path = os.path.join(OUTPUT_DIR, filename)
        save_image(processed_img, save_path)
    print(f"Processed {len(image_paths)} images and saved to '{OUTPUT_DIR}'")
