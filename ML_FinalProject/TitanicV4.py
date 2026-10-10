import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')



df = pd.read_csv('Titanic_train.csv')

df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machines": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Gaussian Naive Bayes": GaussianNB()
}

best_model = None
best_acc = 0

print("--- Model Performance Comparison ---")
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    acc = accuracy_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    
    print(f"{name}: Accuracy = {acc:.2%}, MSE = {mse:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = m
        best_model_name = name

print(f"\nResult: '{best_model_name}' selected for prediction (Highest Accuracy).")
print("="*40)


def get_user_prediction():
    print("\n--- Titanic Survival Prediction System ---")
    try:
        pclass = int(input("Passenger Class (1, 2, or 3): "))
        sex_input = input("Gendor (male/female): ").lower()
        sex = 1 if sex_input == 'female' else 0
        age = float(input("Age: "))
        sibsp = int(input("Number of Siblings/Spouses aboard: "))
        parch = int(input("Number of Parents/Children aboard: "))
        fare = float(input("Ticket Fare: "))

        user_df = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare]], columns=features)
        
        prediction = best_model.predict(user_df)[0]
        probability = best_model.predict_proba(user_df)[0]

        print("\n" + "="*30)
        print(f"Model Used: {best_model_name}")
        if prediction == 1:
            print(f"Outcome: SURVIVED :-)")
            print(f"Survival Probability: {probability[1]:.2%}")
        else:
            print(f"Outcome: NOT SURVIVED :-(")
            print(f"Death Probability: {probability[0]:.2%}")
        print("="*30)

    except ValueError:
        print("\n[Error]: Invalid input! Please enter numerical values.")

get_user_prediction()