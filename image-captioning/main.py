import os
import numpy as np
from PIL import Image
from pickle import dump
import tqdm
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

dataset_images = "Flicker8k_Dataset"
model = Xception(include_top=False, pooling="avg", weights="imagenet")

def extract_features(directory):
    features = {}
    valid_images = {".jpg", ".jpeg", ".png"}
    for img in tqdm.tqdm(os.listdir(directory)):
        ext = os.path.splitext(img)[1].lower()
        if ext not in valid_images:
            continue
        filename = os.path.join(directory, img)
        image = Image.open(filename).convert("RGB")
        image = image.resize((299, 299))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        features[img] = feature[0]
    return features

if __name__ == "__main__":
    features = extract_features(dataset_images)
    with open("features.p", "wb") as f:
        dump(features, f)
    print("Features saved successfully to features.p")
