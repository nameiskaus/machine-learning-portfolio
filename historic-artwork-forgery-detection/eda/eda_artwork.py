import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Load your mapped.csv
df = pd.read_csv('data/mapped.csv')

# 2. Basic info
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print(df.head())

# 3. Missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 4. Uniqueness/counts for each key column
for col in ['artist', 'title', 'period', 'base', 'nationality', 'picture data']:
    print(f"\nColumn: {col}")
    print("Unique values:", df[col].nunique())
    print("Most common values:")
    print(df[col].value_counts().head(10))

# 5. Plot top 20 artists
plt.figure(figsize=(12, 5))
df['artist'].value_counts().head(20).plot(kind='bar')
plt.title("Top 20 Artists by Artwork Count")
plt.ylabel("Number of Artworks")
plt.xlabel("Artist")
plt.tight_layout()
plt.show()

# 6. Plot top 10 nationalities as a pie chart
plt.figure(figsize=(6, 6))
df['nationality'].value_counts().head(10).plot(kind='pie', autopct='%1.1f%%')
plt.title("Top 10 Nationalities")
plt.ylabel("")
plt.show()

# 7. Check for missing images
if 'image_path' in df.columns:
    missing_images = df[~df['image_path'].apply(os.path.exists)]
    print("Missing images:", len(missing_images))

# 8. Optional: Show a few images with metadata (comment out if running in terminal)
try:
    from PIL import Image
    sample = df.sample(3)
    for idx, row in sample.iterrows():
        print(f"\nImage: {row['image_path']}")
        print(f"Artist: {row['artist']}, Title: {row['title']}, Period: {row['period']}, Base: {row['base']}, Nationality: {row['nationality']}")
        img = Image.open(row['image_path'])
        img.show()
except Exception as e:
    print("Couldn't display images (maybe running in terminal or missing PIL):", e)
