import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib  # for saving the model

# Load CSV file
df = pd.read_csv('merged_artwork_metadata.csv')

print(df.head())
print("Columns:", df.columns.tolist())

# Columns to drop safely
drop_cols = ['ID', 'title', 'picture data', 'file info', 'jpg url', 'image_path', 'url', 'base', 'image_exists', 'resolution', 'color_type', 'file_size']
cols_to_drop = [col for col in drop_cols if col in df.columns]
df = df.drop(columns=cols_to_drop)

# Extract birth year from 'born-died' column and convert to float
df['born_year'] = df['born-died'].str.extract(r'\((\d{4})').astype(float)
df = df.drop(columns=['born-died'])
df['born_year'] = df['born_year'].fillna(df['born_year'].mean())

# Encode categorical variables
label_encoders = {}
for col in ['artist', 'period', 'school', 'nationality']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['artist'])
y = df['artist']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification report for test classes only
unique_test_classes = sorted(y_test.unique())
print(classification_report(
    y_test,
    y_pred,
    labels=unique_test_classes,
    target_names=[label_encoders['artist'].inverse_transform([i])[0] for i in unique_test_classes]
))

# Save the trained model to file
joblib.dump(clf, 'random_forest_artist_model.joblib')
print("Model saved to random_forest_artist_model.joblib")
