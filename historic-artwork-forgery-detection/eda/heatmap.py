import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your data
df = pd.read_csv(r'data/mapped.csv')

# 2. Find top 10 artists and top 10 nationalities
top_artists = df['artist'].value_counts().head(10).index
top_nationalities = df['nationality'].value_counts().head(10).index

# 3. Filter the dataframe for only these artists and nationalities
filtered_df = df[df['artist'].isin(top_artists) & df['nationality'].isin(top_nationalities)]

# 4. Create a pivot table: rows=artist, columns=nationality, values=counts
heatmap_data = pd.pivot_table(
    filtered_df, 
    values='ID', 
    index='artist', 
    columns='nationality', 
    aggfunc='count', 
    fill_value=0
)

# 5. Plot the heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Blues")
plt.title("Number of Artworks: Top 10 Artists vs Top 10 Nationalities")
plt.xlabel("Nationality")
plt.ylabel("Artist")
plt.tight_layout()
plt.show()
