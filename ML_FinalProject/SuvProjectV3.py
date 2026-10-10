import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')


df = pd.read_csv('suv_data.csv') 

le = LabelEncoder()
df['Gender_enc'] = le.fit_transform(df['Gender']) # Female=0, Male=1

X = df[['Gender_enc', 'Age', 'EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)
accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\n" + "=" * 45)
print(f"Model Accuracy: {accuracy*100:.2f}%")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print("=" * 45)

print("\nEnter person's details for prediction:")

try:
    gender = input("Gender (Male/Female): ").strip().capitalize()
    age = int(input("Age: "))
    salary = int(input("Estimated Salary: "))

 
    gender_encoded = 1 if gender == "Male" else 0


    user_data = pd.DataFrame({
        'Gender_enc': [gender_encoded],
        'Age': [age],
        'EstimatedSalary': [salary]
    })


    user_scaled = scaler.transform(user_data)


    prediction = model.predict(user_scaled)
    probability = model.predict_proba(user_scaled)[:, 1]

    print("\n" + "="*45)
    if prediction[0] == 1:
        print(f"Result: This person is LIKELY to buy an SUV.")
    else:
        print(f"Result: This person is UNLIKELY to buy an SUV.")
    
    print(f"Purchase Probability: {probability[0]*100:.2f}%")
    print("="*45)

except ValueError:
    print("Error: Please enter valid numbers for Age and Salary.")