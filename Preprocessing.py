def preprocessing(dataframe, save_csv=False, output_data_root=''):
    """
    Ham verileri model tahmini için hazırlar.

    Args:
        dataframe: Kullanıcıdan alınan ham veriler.
        save_csv: Temizlenmiş verilerin CSV dosyası olarak kaydedilip kaydedilmeyeceği (varsayılan: False).
        output_data_root: save_csv True ise temizlenmiş verilerin kaydedileceği yol.
    """
    import numpy as np
    import pandas as pd
    import re
    from datetime import datetime
    from sklearn.preprocessing import StandardScaler

    dataframe.columns = ['Title', 'Address', 'City', 'Price(TL)', 'ListingID', 'ListingDate', 'Brand', 'Series', 'Model',
                         'Year', 'Kilometers', 'GearType', 'FuelType', 'BodyType', 'Color', 'EngineSize', 'EnginePower',
                         'DriveType', 'PaintAndPartsCondition', 'TradeInStatus', 'SellerType', 'VehicleTax(TL)',
                         'TramerCondition']
    
    def clean_text_columns(df):
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).apply(lambda x: x.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip())
        return df

    dataframe = clean_text_columns(dataframe)
    dataframe.drop(['Title', 'ListingID'], axis=1, inplace=True)

    dataframe['Price(TL)'] = dataframe['Price(TL)'].str.replace('.', '', regex=False)  # Remove thousand separators
    dataframe['Price(TL)'] = dataframe['Price(TL)'].str.replace(' TL', '', regex=False)  # Remove ' TL'
    dataframe['Price(TL)'] = dataframe['Price(TL)'].str.replace(',', '.', regex=False)  # Replace comma with dot
    dataframe['Price(TL)'] = pd.to_numeric(dataframe['Price(TL)'], errors='coerce')

    dataframe['ListingDate'] = pd.to_datetime(dataframe['ListingDate'], errors='coerce')
    dataframe['Kilometers'] = pd.to_numeric(dataframe['Kilometers'], errors='coerce')

    def extract_number_fixed(value):
        value = str(value).lower()
        value = value.replace("'", "").replace("cm3", "").replace("cc", "").replace("hp", "")
        nums = re.findall(r'\d+', value)
        if len(nums) == 1:
            return float(nums[0])
        elif len(nums) == 2:
            return round((int(nums[0]) + int(nums[1])) / 2)
        else:
            return np.nan

    dataframe['EngineSize'] = dataframe['EngineSize'].apply(extract_number_fixed)
    dataframe['EnginePower'] = dataframe['EnginePower'].apply(extract_number_fixed)

    dataframe.dropna(subset=['Brand', 'Model', 'Year'], inplace=True)  # Drop rows with critical missing values

    categorical_columns = ['BodyType', 'DriveType', 'FuelType', 'SellerType', 'TradeInStatus']
    for col in categorical_columns:
        mode_value = dataframe[col].mode()[0]
        dataframe[col] = dataframe[col].fillna(mode_value)

    numerical_columns = ['EngineSize', 'EnginePower']
    for col in numerical_columns:
        median_value = dataframe[col].median()
        dataframe[col] = dataframe[col].fillna(median_value)


    def calculate_mtv(engine_size, vehicle_year, price):
        current_year = datetime.now().year
        try:
            vehicle_age = current_year - int(vehicle_year)
            engine_size = float(engine_size)
            price = float(price)
        except ValueError:
            return np.nan

        return 0  

    dataframe['VehicleTax(TL)'] = dataframe.apply(lambda row: calculate_mtv(row['EngineSize'], row['Year'], row['Price(TL)']), axis=1)

    numerical_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        q1 = dataframe[col].quantile(0.01)
        q3 = dataframe[col].quantile(0.98)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        dataframe[col] = np.where(dataframe[col] < lower_bound, lower_bound, dataframe[col])
        dataframe[col] = np.where(dataframe[col] > upper_bound, upper_bound, dataframe[col])

    dataframe['NEW_District'] = dataframe['Address'].apply(lambda x: x.split()[-1])
    dataframe['NEW_DaysSinceListing'] = (datetime.now() - dataframe['ListingDate']).dt.days
    dataframe['NEW_CarAge'] = datetime.now().year - dataframe['Year']

    dataframe.drop(['Address', 'ListingDate'], axis=1, inplace=True)

    dataframe = pd.get_dummies(dataframe, drop_first=True)

    scaler = StandardScaler()
    numerical_features = ['Kilometers', 'EngineSize', 'EnginePower', 'VehicleTax(TL)', 'NEW_DaysSinceListing', 'NEW_CarAge']
    dataframe[numerical_features] = scaler.fit_transform(dataframe[numerical_features])

    if save_csv:
        dataframe.to_csv(output_data_root + "clean_data.csv", index=False)
        print(f"Model saved as {output_data_root}/clean_data.csv")

    return dataframe