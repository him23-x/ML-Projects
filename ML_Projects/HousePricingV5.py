import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')



def load_and_clean_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = [line.strip().strip('"') for line in lines]
    df = pd.read_csv(io.StringIO("\n".join(cleaned_lines)))
    

    if df.columns[0].startswith('Unnamed') or df.columns[0] == '':
        df = df.drop(df.columns[0], axis=1)
    return df


data = load_and_clean_data('Data_For_Model.csv')


core_features = ['square', 'livingRoom', 'drawingRoom', 'bathRoom', 'floor_height', 'ladderRatio', 'elevator_1.0']
building_type_dummies = ['buildingType_2.0', 'buildingType_3.0', 'buildingType_4.0']
structure_dummies = ['buildingStructure_2', 'buildingStructure_3', 'buildingStructure_4', 'buildingStructure_5', 'buildingStructure_6']

all_features = core_features + building_type_dummies + structure_dummies
target = 'totalPrice'


df_model = data[all_features + [target]].dropna()
X = df_model[all_features]
y = df_model[target]

model = LinearRegression()
model.fit(X, y)



def predict_house_price():
    print("\n" + "="*40)
    print("========== House Salary Pridiction ==========")
    
    
    scaling_map = {
        'square': [7.0, 640.0],
        'livingRoom': [0, 9],
        'drawingRoom': [0, 5],
        'bathRoom': [0, 7],
        'floor_height': [1, 50]
    }

    def scale(val, name):
        v_min, v_max = scaling_map[name]
        return (val - v_min) / (v_max - v_min)

    try:
        AreaSquare = float(input("1. Area in square meters (e.g. 100): "))
        LivingRoomNo = float(input("2. Number of bedrooms/living rooms (e.g. 2): "))
        HallsRoomNo = float(input("3. Number of halls/lounges (e.g. 1): "))
        BathroomNo = float(input("4. Number of bathrooms (e.g. 2): "))
        FloorNo = float(input("5. Floor number (e.g. 5): "))
        l_rat = float(input("6. Stair ratio (e.g. 0.2): "))
        elev = int(input("7. Is there an elevator? (Yes: 1 / No: 0): "))

        print("\n========== Building Type ==========")
        print("1: Tower | 2: Bungalow | 3: Combination | 4: Plate")
        b_type = int(input("Choose Number Your Build: "))

        print("\n========== Building Structure ==========")
        print("1: Other | 2: Mixed | 3: Brick-Wood | 4: Brick-Concrete | 5: Steel | 6: Steel-Concrete")
        b_struct = int(input("Choose Number Your Build: "))


        input_data = pd.DataFrame(0, index=[0], columns=all_features)
        

        input_data['square'] = scale(AreaSquare, 'square')
        input_data['livingRoom'] = scale(LivingRoomNo, 'livingRoom')
        input_data['drawingRoom'] = scale(HallsRoomNo, 'drawingRoom')
        input_data['bathRoom'] = scale(BathroomNo, 'bathRoom')
        # input_data['floor'] = scale(FloorNo, 'floor')
        input_data['floor_height'] = scale(FloorNo, 'floor_height')
        input_data['ladderRatio'] = l_rat 
        input_data['elevator_1.0'] = 1 if elev == 1 else 0


        if b_type > 1:
            col = f'buildingType_{float(b_type)}'
            if col in all_features: input_data[col] = 1
        
        if b_struct > 1:
            col = f'buildingStructure_{b_struct}'
            if col in all_features: input_data[col] = 1



        prediction = model.predict(input_data)[0]

        real_price = prediction * 10000

        print("\n========================================")
        print(f"Raw model value: {prediction:.2f}")
        print(f"Predicted full price: {real_price:,.2f}")
        print("Note: The price is measured in Chinese Yuan")
        print("========================================")

    except Exception as e:
        print(f"Please Enter Data Correctly :-(")

if __name__ == "__main__":
    predict_house_price()