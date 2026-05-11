import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision import models
from torchvision.models import ResNet50_Weights
import pandas as pd
import pickle

# ====== CONFIG ======
IMAGE_PATH = input("Enter image path: ").strip()
MODEL_PATH = "models/best_artist_classifier_cnn.pth"
ENCODER_PATH = "models/label_encoder.pkl"
ARTIST_CSV_PATH = "data/final dataset.csv"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== TRANSFORM ======
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ====== LOAD IMAGE ======
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

# ====== LOAD ENCODER ======
with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)

# ====== LOAD MODEL ======
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ====== PREDICT ======
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
    predicted_label = label_encoder.inverse_transform(predicted.cpu().numpy())[0]

print(f"\nPredicted Artist: {predicted_label}")

# ====== LOAD ARTIST DETAILS ======
artist_df = pd.read_csv(ARTIST_CSV_PATH)
artist_df.columns = artist_df.columns.str.strip().str.lower()
predicted_label_clean = predicted_label.strip().lower()

if 'artist' not in artist_df.columns:
    print("ERROR: 'artist' column not found in CSV. Found columns:", artist_df.columns.tolist())
    exit()

artist_row = artist_df[artist_df['artist'].str.strip().str.lower() == predicted_label_clean]

# ====== DISPLAY ARTIST INFO ======
if not artist_row.empty:
    info = artist_row.iloc[0]
    print("\nArtist Details:")
    print(f"Born–Died: {info.get('born-died', 'N/A')}")
    print(f"Period: {info.get('period', 'N/A')}")
    print(f"Nationality: {info.get('nationality', 'N/A')}")
else:
    print("\nNo matching artist details found in CSV.")
    print("Available artists:")
    print(artist_df['artist'].unique())
