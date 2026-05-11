import os
import numpy as np
from pickle import dump, load
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model
import tensorflow as tf

dataset_text = "Flickr8k_text"
dataset_images = "Flicker8k_Dataset"

def load_doc(filename):
    with open(filename, "r") as file:
        return file.read()

def load_photos(filename):
    photos = load_doc(filename).splitlines()
    photos_present = [p for p in photos if os.path.exists(os.path.join(dataset_images, p))]
    return photos_present

def load_clean_descriptions(filename, photos):
    descriptions = {}
    with open(filename, "r") as file:
        for line in file:
            words = line.strip().split()
            if len(words) < 2:
                continue
            image, image_caption = words[0], words[1:]
            if image in photos:
                desc = "<start> " + " ".join(image_caption) + " <end>"
                descriptions.setdefault(image, []).append(desc)
    return descriptions

def load_features(photos):
    all_features = load(open("features.p", "rb"))
    features = {}
    for k in photos:
        if k in all_features:
            feat = all_features[k]
            arr = np.array(feat)
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr[0]
            features[k] = arr.astype("float32")
    return features

def dict_to_list(descriptions):
    return [desc for key in descriptions for desc in descriptions[key]]

def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)

def create_sequences(tokenizer, max_length_val, desc_list, feature, vocab_size):
    X1, X2, y = [], [], []
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length_val)[0]
            out_seq = to_categorical(out_seq, num_classes=vocab_size)
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def data_generator(descriptions, photos, features, tokenizer, max_length_val, vocab_size):
    def generator():
        while True:
            for key in photos:
                desc_list = descriptions[key]
                feature = features[key]
                input_image, input_sequence, output_word = create_sequences(tokenizer, max_length_val, desc_list, feature, vocab_size)
                for i in range(len(input_image)):
                    yield (input_image[i], input_sequence[i]), output_word[i]
    output_signature = (
        (tf.TensorSpec(shape=(2048,), dtype=tf.float32), tf.TensorSpec(shape=(max_length_val,), dtype=tf.int32)),
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.batch(32)
    return dataset

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
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model

if __name__ == "__main__":
    filename = dataset_text + "/Flickr_8k.trainImages.txt"
    train_imgs = load_photos(filename)
    train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
    train_features = load_features(train_imgs)
    tokenizer = create_tokenizer(train_descriptions)
    dump(tokenizer, open("tokenizer.p", "wb"))
    vocab_size = len(tokenizer.word_index) + 1
    max_len = max_length(train_descriptions)
    dump(max_len, open("max_len.p", "wb"))
    model = define_model(vocab_size, max_len)
    epochs = 20
    steps_per_epoch = max(1, len(train_imgs) // 32)
    for i in range(epochs):
        dataset = data_generator(train_descriptions, train_imgs, train_features, tokenizer, max_len, vocab_size)
        model.fit(dataset, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)
        model.save(f"models/model_{i}.h5")
