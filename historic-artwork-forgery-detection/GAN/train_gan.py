"""
train_gan.py
------------
Trains a GAN for artwork forgery/duplication detection.
Expects preprocessed images in the specified input directory.
Saves trained models and generated images to the appropriate folders.
"""



import tensorflow as tf
from tensorflow.keras import layers
from latent_utils import generate_outlier_latent
import os
from PIL import Image, ImageOps
import numpy as np

latent_dim = 100
img_size = 64
batch_size = 1
epochs = 4

def make_generator_model():
    model = tf.keras.Sequential([
        layers.Dense(8*8*256, use_bias=False, input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'),
    ])
    return model

def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Resizing(img_size, img_size),
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[img_size, img_size, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1),
    ])
    return model

def load_image(file_path):
    img = Image.open(file_path)
    img = ImageOps.exif_transpose(img)    # <-- Correct orientation based on EXIF metadata
    img = img.resize((img_size, img_size)).convert("RGB")
    img_array = np.array(img).astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
    return img_array

def load_and_preprocess_dataset(folder, batch_size):
    files = [os.path.join(folder, fname) for fname in os.listdir(folder) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
    dataset = tf.data.Dataset.from_tensor_slices(files)

    def _load(img_path):
        img = tf.numpy_function(load_image, [img_path], tf.float32)
        img.set_shape([img_size, img_size, 3])
        return img

    return dataset.map(_load).shuffle(len(files)).batch(batch_size)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = generate_outlier_latent(batch_size, latent_dim)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def train(dataset, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch)
        print(f"Gen loss: {g_loss.numpy():.4f}, Disc loss: {d_loss.numpy():.4f}")

if __name__ == "__main__":
    dataset = load_and_preprocess_dataset("data/GAN-processed", batch_size=batch_size)
    train(dataset, epochs)
