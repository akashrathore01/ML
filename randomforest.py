import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
print("Step 1: Load dataset")
df = pd.read_csv('matches.csv')
df.head()
print("Step 2: Select features")
features = ['city', 'venue', 'team1', 'team2']
target = 'winner'
features, target
print("Step 3: Encoding")
le_dict = {}
for col in features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
print("Step 4: Prepare X and y")
X = df[features]
target_le = LabelEncoder()
y = target_le.fit_transform(df[target])
print("Step 5: Train model")
model = RandomForestClassifier(n_estimators=200 ,random_state=42)
model.fit(X, y)
print("Model trained")


city = "Bangalore"
venue = "M Chinnaswamy Stadium"
team1 = "Royal Challengers Bangalore"
team2 = "Chennai Super Kings"

city_encoded = le_dict['city'].transform([city])[0]
venue_encoded = le_dict['venue'].transform([venue])[0]
team1_encoded = le_dict['team1'].transform([team1])[0]
team2_encoded = le_dict['team2'].transform([team2])[0]


input_data = [[city_encoded, venue_encoded, team1_encoded, team2_encoded]]


prediction = model.predict(input_data)


winner = target_le.inverse_transform(prediction)

print("Predicted Winner:", winner[0])
print("Step 6: Save model")
joblib.dump({
    'pipeline': model,
    'label_encoders': le_dict,
    'feature_names': features
}, 'ipl_model.pkl')
print("Model saved")
