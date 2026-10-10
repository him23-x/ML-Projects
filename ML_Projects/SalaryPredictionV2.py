import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


df = pd.read_csv('Salary Data.csv')

df = df.dropna()


X = df[['Age', 'Gender', 'Education Level', 'Years of Experience']]
y = df['Salary']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)



preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            ['Gender', 'Education Level']
        )
    ],
    remainder='passthrough'
)



model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])



model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)


rmse = mse ** 0.5


mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)


print("\n=== Model Performance ===")

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

print(f"R2 Score: {r2:.4f}")

print(f"Accuracy (R2%): {r2 * 100:.2f}%")


print("\n--- Predict New Salary ---")

age = float(input("Enter Age: "))

gender = input(
    "Enter Gender (Male/Female): "
).strip().title()

education = input(
    "Enter Education (High School/Bachelor's/Master's/PhD): "
).strip()

experience = float(
    input("Enter Years of Experience: ")
)


new_data = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'Education Level': education,
    'Years of Experience': experience
}])


prediction = model.predict(new_data)

print(f"\nEstimated Salary: ${prediction[0]:,.2f}")