import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("House_price.csv")


data = data.dropna()
#df.fillna(df.mean(), inplace=True)
##data = data.drop_duplicates()

X = data.drop(columns = ["Address","Price"])
Y = data["Price"]

xTrain,xTest,yTrain, yTest = train_test_split(X,Y, test_size=0.3, random_state=42)



LR = LinearRegression()
LR.fit(xTrain,yTrain)


Predicted = LR.predict(xTest)



print("=============Accuracy=============")
print(LR.score(xTrain,yTrain))
print(LR.score(xTest,yTest))




print("=============coefficients=============")
print(LR.coef_)
print(LR.intercept_)



print("=============Pridiction=============")

avgArea = float(input("Enter [AVG Area]: "))
HouseAge = float(input("Enter [House Age]: "))
NoRooms = float(input("Enter [No Rooms]: "))
NoBedrooms = float(input("Enter [No Bedrooms]: "))
AreaPopulation = float(input("Enter [Area Population]: "))

HousePredict = pd.DataFrame([[avgArea, HouseAge, NoRooms, NoBedrooms, AreaPopulation]], 
                             columns=X.columns)

prediction = LR.predict(HousePredict)
print(f"Predicted Price: {prediction[0]}")
