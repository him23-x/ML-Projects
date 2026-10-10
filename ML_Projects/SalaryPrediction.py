import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('Salary Data.csv')
df = df.dropna()


education_map = {
    "High School": 0,
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3
}
df["Education Level"] = df["Education Level"].map(education_map)


gender_map = {"Male": 0, "Female": 1}
df["Gender"] = df["Gender"].map(gender_map)


df = df.dropna()

X = df[['Age', 'Gender', 'Education Level', 'Years of Experience']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")


print("\n--- Predict New Salary ---")
age = float(input("Enter Age: "))
gender = input("Enter Gender (Male/Female): ")
edu = input("Enter Education (High School/Bachelor's/Master's/PhD): ")
exp = float(input("Enter Years of Experience: "))


gender_val = gender_map.get(gender, 0)
edu_val = education_map.get(edu, 0)


new_data = pd.DataFrame([[age, gender_val, edu_val, exp]], 
                        columns=['Age', 'Gender', 'Education Level', 'Years of Experience'])
prediction = model.predict(new_data)
print(f"\nEstimated Salary: ${prediction[0]:,.2f}")