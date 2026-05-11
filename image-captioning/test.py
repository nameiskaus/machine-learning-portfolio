import sys
import os
import numpy as np
from PIL import Image
from pickle import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="Path to the image file")
    args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args(args=[])
    return args.image if args.image else "dog.jpg"

def extract_features(filename, feature_model):
    if not os.path.exists(filename):
        print(f"Image file '{filename}' not found.")
        return None
    try:
        image = Image.open(filename).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
    image = image.resize((299, 299))
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    feat = feature_model.predict(image, verbose=0)
    return feat[0]

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length_val):
    in_text = "<start>"
    for _ in range(max_length_val):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length_val)
        yhat = model.predict([np.expand_dims(photo, 0), sequence], verbose=0)
        yhat = np.argmax(yhat[0])
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        if word == "<end>":
            break
        in_text += " " + word
    return " ".join([w for w in in_text.split() if w not in ("<start>", "<end>")])

def define_model(vocab_size, max_length_val):
    inputs1 = Input(shape=(2048,), name="input_1")
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation="relu")(fe1)

    inputs2 = Input(shape=(max_length_val,), name="input_2")
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation="relu")(decoder1)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

if __name__ == "__main__":
    img_path = parse_arguments()
    tokenizer = load(open("tokenizer.p", "rb"))
    max_len = load(open("max_len.p", "rb"))
    vocab_size = len(tokenizer.word_index) + 1

    model = define_model(vocab_size, max_len)
    model.load_weights("models/model_19.h5")

    feature_extractor = Xception(include_top=False, pooling="avg", weights="imagenet")
    photo = extract_features(img_path, feature_extractor)

    if photo is not None:
        description = generate_desc(model, tokenizer, photo, max_len)
        print(description)
