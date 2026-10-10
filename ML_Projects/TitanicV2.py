import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv('Titanic_train.csv')


df['Age'] = df['Age'].fillna(df['Age'].median())


sex_map = {"male": 0, "female": 1}
df["Sex"] = df["Sex"].map(sex_map)


X = df.drop(columns = ["PassengerId", "Survived", "Name", "Ticket", "Cabin", "Embarked"])
Y = df["Survived"]



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)



model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)



print("===============Accuracy===============")
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")


print("===============Pridiction===============")
isServived = pd.DataFrame([{
    "Pclass" : 1,
    "Sex" : 0,
    "Age" : 30,
    "SibSp" : 1,
    "Parch" : 0,
    "Fare" : 70
}])

prediction = model.predict(isServived)
probability = model.predict_proba(isServived)

print(f"The Prediction         : {'Survived' if prediction[0] == 1 else 'Not Survived'}")
print(f"Probability of survival: {probability[0][1]:.2%}")
print(f"Possibility of death   : {probability[0][0]:.2%}")