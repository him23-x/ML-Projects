import requests
import pandas as pd

url = "https://dlfjdlfj.com"

response = requests.get(url)

data = pd.DataFrame(response.json())

df = pd.DataFrame(response.json())

df.isnull().sum()

df.fillna(df.mean())

df.dropna()

df.drop_duplicates()
df.astype(int)


from sklearn.preprocessing import LabelEncorder

le = LabelEncorder()

df["Gendor"] = le.fit_transform(df["Gendor"])

pd.get_dummies(df,columns= df["Gendor"])




from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df["salary","Age"] = scaler.fit_transform(df["salary","Age"])





from sklearn.model_selection import train_test_split

x = train_test_split(x,y,test_size=0.3,random_state= 42)







import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression 

x = np.array([3,25,5,2,9]).reshape(-1,1)
y = np.array([23,69,35,9,99])

model = LinearRegression()
model.fit(x,y)





plt.scatter(x,y)

plt.plot(x,model.predict(x))
plt.show()


