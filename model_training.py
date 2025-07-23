import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import joblib
import os

# Load dataset
df = pd.read_csv('car_data.csv')

# Drop Car_Name
df.drop('Car_Name', axis=1, inplace=True)

# Rename to match consistent casing
df.rename(columns={'Driven_kms': 'Driven_Kms', 'Selling_type': 'Seller_Type'}, inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])
df['Seller_Type'] = le.fit_transform(df['Seller_Type'])
df['Transmission'] = le.fit_transform(df['Transmission'])

# Define features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"R2 Score: {score:.2f}")

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/car_price_model.pkl')
print("Model saved to models/car_price_model.pkl")
