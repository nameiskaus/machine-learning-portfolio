import pandas as pd
import numpy as np
import os
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- 1. Load your mapped DataFrame ---
df = pd.read_csv('complete/mapped.csv')

# --- 2. Use the first 10,000 images ---
df = df.iloc[:10000].reset_index(drop=True)

# --- 3. Encode artist labels ---
le = LabelEncoder()
df['artist_encoded'] = le.fit_transform(df['artist'])
num_classes = len(le.classes_)

# --- 4. Train/Validation Split (stratified) ---
train_df, val_df = train_test_split(
    df, 
    test_size=0.2, 
    stratify=df['artist_encoded'], 
    random_state=42
)

# --- 5. Data Generator ---
class ArtDataset(Sequence):
    def __init__(self, df, batch_size=32, img_size=(224,224), shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_df = self.df.iloc[batch_indexes]
        images, labels = [], []
        for _, row in batch_df.iterrows():
            img = load_img(row['image_path'], target_size=self.img_size)
            img = img_to_array(img)
            img = preprocess_input(img)
            images.append(img)
            labels.append(row['artist_encoded'])
        return np.array(images), to_categorical(labels, num_classes=num_classes)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# --- 6. Build the Model ---
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# --- 7. Compile ---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 8. Create Datasets ---
batch_size = 32
train_gen = ArtDataset(train_df, batch_size=batch_size)
val_gen = ArtDataset(val_df, batch_size=batch_size, shuffle=False)

# --- 9. Train ---
model.fit(train_gen, validation_data=val_gen, epochs=10)

# --- 10. Save Model ---
model.save('artist_classifier_resnet50_10k_first.h5')

# --- 11. Predict Function Example ---
def predict_artist(img_path):
    img = load_img(img_path, target_size=(224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    pred = model.predict(np.expand_dims(img, axis=0))
    class_idx = np.argmax(pred)
    artist = le.inverse_transform([class_idx])[0]
    return artist

# Example usage (after training):
# print(predict_artist(val_df.iloc[0]['image_path']))
