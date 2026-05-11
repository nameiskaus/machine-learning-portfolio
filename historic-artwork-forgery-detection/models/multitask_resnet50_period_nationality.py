import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
# 1. Load your data

df = pd.read_csv('./merged_artwork_metadata.csv')


# Filter to only keep rows where the image file exists
df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
print(f"Images with files present: {len(df)}")

# Now sample for testing
df = df.sample(500, random_state=42).reset_index(drop=True)

# 2. Encode labels as integers
from sklearn.preprocessing import LabelEncoder
period_encoder = LabelEncoder()
nationality_encoder = LabelEncoder()
df['period_idx'] = period_encoder.fit_transform(df['period'])
df['nationality_idx'] = nationality_encoder.fit_transform(df['nationality'])

# 3. Define PyTorch Dataset

class ArtDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row['image_path']).convert('RGB')
        except Exception as e:
            # Return the next item instead
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            image = self.transform(image)
        period = row['period_idx']
        nationality = row['nationality_idx']
        return image, torch.tensor(period), torch.tensor(nationality)


# 4. Data transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

dataset = ArtDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 5. Define multitask model (ResNet50 backbone)
num_periods = df['period_idx'].nunique()
num_nationalities = df['nationality_idx'].nunique()

class MultiTaskResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V2')
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.period_head = nn.Linear(in_features, num_periods)
        self.nat_head = nn.Linear(in_features, num_nationalities)
    
    def forward(self, x):
        features = self.backbone(x)
        period_out = self.period_head(features)
        nat_out = self.nat_head(features)
        return period_out, nat_out

model = MultiTaskResNet()

# 6. Loss, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 7. Training loop (simple, single epoch for demo)
for images, period_labels, nat_labels in dataloader:
    period_pred, nat_pred = model(images)
    loss1 = criterion(period_pred, period_labels)
    loss2 = criterion(nat_pred, nat_labels)
    loss = loss1 + loss2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item():.4f}")

print("Done (demo run). For real training, use more data, validation, and multiple epochs.")



