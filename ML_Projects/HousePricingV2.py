import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. تحميل البيانات وتجهيزها (Data Preprocessing)
try:
    df = pd.read_csv('house_prices.csv')
    
    # حذف عمود HouseID لأنه لا يؤثر إحصائياً على السعر
    data = df.drop(columns=['HouseID'])
    
    # التعامل مع القيم المفقودة (إن وجدت)
    data = data.dropna()
    
    # تحويل عمود الموقع (Location) إلى قيم رقمية (One-Hot Encoding)
    # ملاحظة: drop_first=True لتفادي مشكلة التكرار الإحصائي
    data_encoded = pd.get_dummies(data, columns=['Location'], drop_first=True)
    
    # تحديد المدخلات (X) والمخرج (y)
    X = data_encoded.drop(columns=['Price'])
    y = data_encoded['Price']
    
except:
    print("Error in Reading dataset!!")
    exit()

# 2. تقسيم البيانات (80% تدريب، 20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. بناء وتدريب نموذج الانحدار الخطي المتعدد
model = LinearRegression()
model.fit(X_train, y_train)

# 4. تفسير المعاملات (Interpret Coefficients)
coeffs = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient Value'])
print("\n--- تفسير المعاملات (Coefficients) ---")
print(coeffs)
print(f"Intercept: {model.intercept_:.2f}")

# 5. تقييم الموديل باستخدام R² Score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("\n--- تقييم الموديل ---")
print(f"R-squared Score: {r2:.4f}")

# 6. التنبؤ بسعر منزل جديد (Predict)
# سنقوم بإنشاء بيانات لمنزل جديد بنفس هيكلية الأعمدة المدربة
new_house_data = {col: [0] for col in X.columns}
new_house_df = pd.DataFrame(new_house_data)

# تعبئة مواصفات المنزل الجديد
new_house_df['Bedrooms'] = 3
new_house_df['Bathrooms'] = 2
new_house_df['SquareFeet'] = 2000
# إذا أردت اختيار مدينة معينة (مثلاً New York)، نجعل العمود الخاص بها 1
if 'Location_New York' in new_house_df.columns:
    new_house_df['Location_New York'] = 1

predicted_price = model.predict(new_house_df)[0]
print("\n--- التوقع ---")
print(f"السعر المتوقع للمنزل الجديد هو: ${predicted_price:,.2f}")