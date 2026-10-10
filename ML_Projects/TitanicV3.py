import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Data Preprocessing 
# ==========================================

# Load the dataset
df = pd.read_csv('Titanic_train.csv')

# Accuracy Improvement: Fill missing Age based on Sex and Pclass group medians
df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

# Mapping Gender to numbers
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Filling missing Fare and Embarked values
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna('S')

# Feature Selection
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
X = df[features]
y = df["Survived"]

# Split data into training and testing (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. Model Building & Training
# ==========================================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==========================================
# 3. Evaluation & MSE Calculation
# ==========================================

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("="*40)
print(f"Model Accuracy: {accuracy:.2%}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print("="*40)

# ==========================================
# 4. Interactive User Input
# ==========================================

def get_user_prediction():
    print("\n--- Titanic Survival Prediction System ---")
    print("Please enter the passenger details below:")
    try:
        pclass = int(input("Passenger Class (1, 2, or 3): "))
        sex_input = input("Sex (male/female): ").lower()
        sex = 1 if sex_input == 'female' else 0
        age = float(input("Age: "))
        sibsp = int(input("Number of Siblings/Spouses aboard: "))
        parch = int(input("Number of Parents/Children aboard: "))
        fare = float(input("Ticket Fare (e.g., 50.0): "))

        # Convert input to DataFrame
        user_df = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare]], columns=features)
        
        # Prediction and Probabilities
        prediction = model.predict(user_df)[0]
        probability = model.predict_proba(user_df)[0]

        print("\n" + "="*30)
        if prediction == 1:
            print(f"Outcome: SURVIVED ✅")
            print(f"Survival Probability: {probability[1]:.2%}")
        else:
            print(f"Outcome: NOT SURVIVED ❌")
            print(f"Death Probability: {probability[0]:.2%}")
        print("="*30)

    except ValueError:
        print("\n[Error]: Invalid input! Please enter numerical values where required.")

# Run the interactive session
get_user_prediction()