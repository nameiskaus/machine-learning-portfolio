import pandas as pd

df = pd.read_csv(r'data/merged_artwork_metadata.csv')
print(df.columns.tolist())

import matplotlib.pyplot as plt

# 1. Quick overview
print(f"Rows: {len(df)}")
print("Columns:", df.columns.tolist())
print(df.head())

# 2. Missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 3. Distribution for key columns
key_columns = ['artist', 'title', 'period', 'base', 'nationality']
for col in key_columns:
    print(f"\nColumn: {col}")
    print("Unique values:", df[col].nunique())
    print("Most common values:")
    print(df[col].value_counts().head(10))

# 4. Plot top 10 periods
df['period'].value_counts().head(10).plot(kind='bar', figsize=(8,4), color='skyblue')
plt.title("Top 10 Art Periods")
plt.xlabel("Period")
plt.ylabel("Number of Artworks")
plt.tight_layout()
plt.show()

# 5. Plot top 10 bases
df['base'].value_counts().head(10).plot(kind='bar', figsize=(8,4), color='lightgreen')
plt.title("Top 10 Bases")
plt.xlabel("Base Location")
plt.ylabel("Number of Artworks")
plt.tight_layout()
plt.show()

# 6. Nationality pie chart
df['nationality'].value_counts().head(10).plot(
    kind='pie', autopct='%1.1f%%', figsize=(6,6), colormap='tab20')
plt.title("Top 10 Nationalities")
plt.ylabel("")
plt.show()

for col in ['new_column_1', 'new_column_2']:
    print(f"\nColumn: {col}")
    print("Unique values:", df[col].nunique())
    print("Most common values:")
    print(df[col].value_counts().head(10))
