# 🚗 Car Price Prediction using Machine Learning

This project predicts the selling price of used cars based on key features like present price, driven kilometers, fuel type, transmission type, and ownership. It uses a Random Forest Regressor model trained on real-world data.

---

## 📊 Dataset Columns
- `Car_Name` – Name of the car (dropped during training)
- `Year` – Year of purchase
- `Selling_Price` – Selling price of the car (target)
- `Present_Price` – Price of the car when bought
- `Driven_kms` – Kilometers driven
- `Fuel_Type` – Petrol / Diesel / CNG
- `Selling_type` – Individual / Dealer
- `Transmission` – Manual / Automatic
- `Owner` – Number of previous owners

---

## 🔧 Technologies Used
- **Python**
- **Pandas & NumPy** for data preprocessing
- **Scikit-learn** for model training
- **Random Forest Regressor**
- **Streamlit** for UI
- **Joblib** for model serialization

---

## 📈 Model Training
- Categorical features encoded using `LabelEncoder`
- Dataset split into training and test sets
- Model evaluated using **R² Score**
- Trained model saved as `car_price_model.pkl`

---

## 🌐 Streamlit App
Users can enter the features of a used car, and the model predicts the selling price instantly.

To run the app:
```bash
streamlit run app.py
📦 Folder Structure
Copy code
carpriceprediction/
├── car_data.csv
├── model_training.py
├── car_price_model.pkl
├── app.py
├── requirements.txt
└── README.md
✅ Project Highlights
Real-world dataset cleaning and preprocessing

Machine learning model selection and training

Performance evaluation and model saving

Frontend deployed using Streamlit

