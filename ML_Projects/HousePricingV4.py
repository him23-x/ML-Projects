import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error # استيراد مكتبة الـ MSE

try:
    data = pd.read_csv("House_price.csv")
    data = data.dropna()
except:
    print("File not found!")
    exit()

X = data.drop(columns = ["Address","Price"])
Y = data["Price"]

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.3, random_state=42)

LR = LinearRegression()
LR.fit(xTrain, yTrain)

Predicted = LR.predict(xTest)

# 1- تعديل طباعة الـ Accuracy لنسبة مئوية
print("\n" + "="*13 + " Accuracy " + "="*13)
print(f"Train Accuracy: {LR.score(xTrain, yTrain)*100:.2f}%")
print(f"Test Accuracy:  {LR.score(xTest, yTest)*100:.2f}%")

# 3- حساب الـ MSE والـ RMSE (للتاكد من جودة الموديل)
mse = mean_squared_error(yTest, Predicted)
rmse = np.sqrt(mse) 
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root MSE (RMSE): {rmse:.2f}  <-- (متوسط الخطأ في السعر بالدولار)")

print("\n" + "="*13 + " Prediction " + "="*13)
avgArea = float(input("Enter [AVG Area]: "))
HouseAge = float(input("Enter [House Age]: "))
NoRooms = float(input("Enter [No Rooms]: "))
NoBedrooms = float(input("Enter [No Bedrooms]: "))
AreaPopulation = float(input("Enter [Area Population]: "))

HousePredict = pd.DataFrame([[avgArea, HouseAge, NoRooms, NoBedrooms, AreaPopulation]], 
                             columns=X.columns)

raw_prediction = LR.predict(HousePredict)[0]

# 2- حل مشكلة القيمة السالبة باستخدام max(0, ...)
final_prediction = max(0, raw_prediction)

print(f"\nPredicted Price: ${final_prediction:,.2f}")
if final_prediction == 0 and raw_prediction < 0:
    print("Note: The inputs are too low for this model's logic, resulting in a $0 estimate.")