import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. تحميل البيانات (بافتراض اسم الملف house_data.csv)
try:
    df = pd.read_csv('house_data.csv')
    
    # 2. معالجة البيانات (Handling Missing Values)
    # سنقوم بملء الخانات الفارغة بمتوسط القيم في كل عمود
    df.fillna(df.mean(), inplace=True)
    
    # تحديد المدخلات (المساحة، الغرف، الموقع) والمخرج (السعر)
    X = df[['Size', 'Rooms', 'Location_Score']]
    y = df['Price']
    
except FileNotFoundError:
    print("يرجى التأكد من وجود ملف house_data.csv")
    exit()

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. تطبيق Multiple Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 4. تفسير المعاملات (Interpret Coefficients)
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("--- Coefficients Interpretation ---")
print(coefficients)
print(f"Intercept (Base Price): {model.intercept_:.2f}")

# 5. تقييم الموديل باستخدام R² Score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\n--- Model Evaluation ---")
print(f"R² Score: {r2:.4f}")

# 6. التنبؤ بسعر منزل جديد
# مثال: مساحة 150 متر، 3 غرف، تقييم موقع 8
new_house = [[150, 3, 8]]
predicted_price = model.predict(new_house)
print(f"\n--- Prediction ---")
print(f"Predicted Price for the new house: ${predicted_price[0]:,.2f}")