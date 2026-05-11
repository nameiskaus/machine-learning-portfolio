import streamlit as st
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import pickle
import pandas as pd
import os

# ==== CONFIG ====
MODEL_PATH = "models/best_artist_classifier_cnn.pth"
ENCODER_PATH = "models/label_encoder.pkl"
ARTIST_CSV_PATH = "data/final dataset.csv"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== LOAD MODEL ====
@st.cache_resource(show_spinner=False)
def load_model():
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 16)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ==== LOAD ENCODER ====
@st.cache_resource(show_spinner=False)
def load_label_encoder():
    with open(ENCODER_PATH, 'rb') as f:
        return pickle.load(f)

# ==== LOAD ARTIST DATA ====
@st.cache_data(show_spinner=False)
def load_artist_data():
    df = pd.read_csv(ARTIST_CSV_PATH)
    df.columns = df.columns.str.strip().str.lower()
    return df

# ==== TRANSFORM ====
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_artist(image: Image.Image, model, label_encoder, artist_df):
    img_t = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_t)
        _, predicted = torch.max(output, 1)
        predicted_label = label_encoder.inverse_transform(predicted.cpu().numpy())[0]

    predicted_label_clean = predicted_label.strip().lower()
    artist_row = artist_df[artist_df['artist'].str.strip().str.lower() == predicted_label_clean]

    info = {}
    if not artist_row.empty:
        info = {
            "Born–Died": artist_row.iloc[0].get('born-died', 'N/A'),
            "Period": artist_row.iloc[0].get('period', 'N/A'),
            "Nationality": artist_row.iloc[0].get('nationality', 'N/A')
        }
    return predicted_label, info

# ==== STREAMLIT UI ====
st.title("Artist Classifier 🎨")
st.write("Upload a painting image and get the predicted artist along with artist details.")

uploaded_file = st.file_uploader("Drag and drop an image here", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        model = load_model()
        label_encoder = load_label_encoder()
        artist_df = load_artist_data()

        predicted_artist, artist_info = predict_artist(image, model, label_encoder, artist_df)

        st.markdown(f"### Predicted Artist: {predicted_artist}")

        if artist_info:
            st.markdown("#### Artist Details:")
            st.write(f"**Born–Died:** {artist_info.get('Born–Died', 'N/A')}")
            st.write(f"**Period:** {artist_info.get('Period', 'N/A')}")
            st.write(f"**Nationality:** {artist_info.get('Nationality', 'N/A')}")
        else:
            st.write("No additional artist details found.")
    except Exception as e:
        st.error(f"Error processing image: {e}")
