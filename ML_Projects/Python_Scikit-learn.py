import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


try:
    df = pd.read_csv('score_updated.csv')
except:
    print("Error in reading dataset")
    exit()


df.dropna(inplace=True)


X = df[['Hours']] 
y = df['Scores']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


Hours = float(input("Enter Number of Hours : "))

hours_to_predict = [[Hours]]
predicted_score = model.predict(hours_to_predict)

print(f"The Prediction is : {predicted_score[0]:.2f}")


plt.scatter(X, y, color='blue', label='Actual Data (from CSV)')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title('Study Hours vs Exam Score (Updated Data)')
plt.xlabel('Hours')
plt.ylabel('Score')
plt.legend()
plt.show()