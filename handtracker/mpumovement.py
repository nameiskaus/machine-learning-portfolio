import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from datetime import datetime
import joblib

data = pd.read_csv('sensor_data.csv')
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%H:%M:%S')
data['time_diff'] = data['Timestamp'].diff().dt.total_seconds().fillna(0)

data['magnitude_t'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
data['magnitude_t1'] = data['magnitude_t'].shift(-1)
data['x_t1'] = data['x'].shift(-1)
data['y_t1'] = data['y'].shift(-1)
data['z_t1'] = data['z'].shift(-1)
data['time_diff_t1'] = data['time_diff'].shift(-1)

data['rel_x'] = data['x'] - data['x'].shift(1)
data['rel_y'] = data['y'] - data['y'].shift(1)
data['rel_z'] = data['z'] - data['z'].shift(1)
data['rel_x_t1'] = data['x_t1'] - data['x']
data['rel_y_t1'] = data['y_t1'] - data['y']
data['rel_z_t1'] = data['z_t1'] - data['z']

data.dropna(inplace=True)

X = data[['x', 'y', 'z', 'time_diff', 'x_t1', 'y_t1', 'z_t1', 'time_diff_t1',
          'rel_x', 'rel_y', 'rel_z', 'rel_x_t1', 'rel_y_t1', 'rel_z_t1']]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, 'movement_predictor.pkl')

def predict_movement(timestamp1, x1, y1, z1, timestamp2, x2, y2, z2):
    model = joblib.load('movement_predictor.pkl')

    t1 = datetime.strptime(timestamp1, '%H:%M:%S')
    t2 = datetime.strptime(timestamp2, '%H:%M:%S')
    time_diff = (t2 - t1).total_seconds()

    rel_x = x2 - x1
    rel_y = y2 - y1
    rel_z = z2 - z1

    input_data = [x1, y1, z1, time_diff, x2, y2, z2, time_diff, rel_x, rel_y, rel_z, rel_x, rel_y, rel_z]
    prediction = model.predict([input_data])
    return prediction[0]

timestamp1 = "20:47:47"
x1, y1, z1 = -9279, -3021, 922
timestamp2 = "20:47:48"
x2, y2, z2 = 12943,-767, -836

movement = predict_movement(timestamp1, x1, y1, z1, timestamp2, x2, y2, z2)
print(f"Predicted movement: {movement}")
