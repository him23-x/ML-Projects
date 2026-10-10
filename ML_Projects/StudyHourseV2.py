import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 

try:
    df = pd.read_csv('score_updated.csv')
    df.dropna(inplace=True) 
    X = df[['Hours']]
    y = df['Scores']
except:
    print("The File doesn't exist:-(")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


print("=============== Accuracy ===============")
print(f"Train Accuracy: {model.score(X_train, y_train)*100:.2f}%")
print(f"Test Accuracy:  {model.score(X_test, y_test)*100:.2f}%")

m = model.coef_[0]
b = model.intercept_    
print(f"\n=============== Equation of the Regression Line ===============")
print(f"y = {m:.2f}x + {b:.2f}")

hours = float(input("\nEnter number of hours: "))

hours_df = pd.DataFrame([[hours]], columns=['Hours'])
prediction = model.predict(hours_df)

print(f"\n=============== Prediction ===============")
print(f"Predicted score for {hours} hours: {prediction[0]:.2f}")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)


print(f"\n=============== Model Evaluation ===============")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {mse**0.5:.2f}") # إضافة للتوضيح

plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.title('Relationship: Study Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.legend()
plt.show()