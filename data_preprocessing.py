import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from data_loader import data_loader

# Remove rows with missing values
def remove_missing_values(data):
    data = data.dropna()

    return data


# Remove rows with duplicates
def remove_duplicates(data: str) -> str:
    data = data.drop_duplicates()

    return data


# Preprocessing "Model" variable
def cleaning_model_var(data):

    # Separates "Model" variable into 2 separate variables - car model number & year manufactured
    data[['Model','Year']] = data.Model.str.split(",", expand=True)

    # Repositioning dataframe columns
    cols = data.columns.tolist()
    cols = cols[:2] + cols[-1:] + cols[2:-1]
    data = data[cols]

    return data


# Preprocessing "Color" variable
def cleaning_color_var(data):

    # Removing "Color" variable from dataset
    data = data.drop('Color', axis=1)

    return data


# Preprocessing "Temperature" variable
def cleaning_temperature_var(data):

    # Separates "Temperature" variable into 2 separate variables - Temperature & Temperature Scale
    data[['Temperature','Temperature Scale']] = data.Temperature.str.split(expand=True)

    # Change "Temperature" data type to continuous numeric
    data["Temperature"] = data["Temperature"].astype("float64")

    # Repositioning dataframe columns
    cols = data.columns.tolist()
    cols = cols[:4] + cols[-1:] + cols[4:-1]
    data = data[cols]

    # Change "Temperature Scale" of sample from °C to °F
    data.iloc[4,:] = data.iloc[4,:].replace("°C", "°F")

    # Covert temperature from °F to °C
    data.loc[data["Temperature Scale"]=="°F", 'Temperature'] = (data[data["Temperature Scale"]=="°F"]["Temperature"] - 32) * 5/9

    # Removing "Temperature Scale" variable since all "Temperature" values are now on the same scale
    data = data.drop(['Temperature Scale'], axis=1)

    return data


# Preprocessing "Factory" variable
def cleaning_factory_var(data):

    # Removing Regions from the "Factory" variable, leaving only Country
    data["Factory"] = data["Factory"].map(lambda x: x.split(", ")[1])

    return data


# Preprocessing "RPM" variable
def cleaning_rpm_var(data):

    # Removing outlying rows with negative RPM
    data = data[data["RPM"]>=0]

    # Applying log transformation to RPM variable
    data["RPM"] = np.log(data["RPM"])

    return data


# Preprocessing "Fuel consupmtion" variable
def cleaning_fuel_consumption_var(data):

    # Applying log transformation to Fuel consumption variable
    data["Fuel consumption"] = np.log(data["Fuel consumption"])

    return data


# Combining car failure types into a single binary target variable
def cleaning_target_variables(data):

    # Assign value of 1 to samples with occurrence of a car failure
    def get_y(row):
        for c in data[['Failure A','Failure B','Failure C','Failure D','Failure E']].columns:
            if row[c]==1:
                return 1

    # Applying get_y to data
    data["y"] = data[["Failure A","Failure B","Failure C","Failure D","Failure E"]].apply(get_y, axis=1)

    # Replacing null values (for data points without car failures) in data with 0
    data["y"] = data["y"].fillna(0)

    return data


# Apply integer encoding to categorical variables
def encode_categorical_data(data):

    # Applying integer encoding on "Model" variable
    model_data = data["Model"]
    model_values = np.array(model_data)
    model_label_encoder = LabelEncoder()
    model_integer_encoded = model_label_encoder.fit_transform(model_values)
    data["model_encoded"] = model_integer_encoded

    # Applying integer encoding on "Year" variable
    year_data = data["Year"]
    year_values = np.array(year_data)
    year_label_encoder = LabelEncoder()
    year_integer_encoded = year_label_encoder.fit_transform(year_values)
    data["year_encoded"] = year_integer_encoded

    # Applying integer encoding on "Usage" 
    usage_data = data["Usage"]
    usage_values = np.array(usage_data)
    usage_label_encoder = LabelEncoder()
    usage_integer_encoded = usage_label_encoder.fit_transform(usage_values)
    """
    Since Low, Medium and High is currently encoded to 1, 2 and 0 respectively, 
    we will update the encoded values to 0, 1 and 2 for Low, Medium and High respectively 
    to ensure natural ordinal relationship between them.
    """
    usage_integer_encoded = usage_integer_encoded.tolist()
    usage_integer_encoded = [3 if item == 2 else item for item in usage_integer_encoded]
    usage_integer_encoded = [2 if item == 0 else item for item in usage_integer_encoded]
    usage_integer_encoded = [0 if item == 1 else item for item in usage_integer_encoded]
    usage_integer_encoded = [1 if item == 3 else item for item in usage_integer_encoded]
    data["usage_encoded"] = usage_integer_encoded

    # Applying integer encoding on "Membership" variable
    membership_data = data["Membership"]
    membership_values = np.array(membership_data)
    membership_label_encoder = LabelEncoder()
    membership_integer_encoded = membership_label_encoder.fit_transform(membership_values)
    data["membership_encoded"] = membership_integer_encoded

    # Applying one-hot encoding on "Factory" variable
    factory_data = data["Factory"]
    factory_values = np.array(factory_data)
    onehot_encoder = OneHotEncoder(sparse=False)
    factory_values = factory_values.reshape(-1, 1)
    onehot_encoded = onehot_encoder.fit_transform(factory_values)
    factory_df = pd.DataFrame(onehot_encoded, columns=["factory_china_encoded","factory_germany_encoded","factory_us_encoded"])
    data = data.reset_index(drop=True)
    data = pd.concat([data,factory_df], axis=1)

    return data

def main():
    data = data_loader()
    data = remove_missing_values(data)
    data = remove_duplicates(data)
    data = cleaning_model_var(data)
    data = cleaning_color_var(data)
    data = cleaning_temperature_var(data)
    data = cleaning_factory_var(data)
    data = cleaning_rpm_var(data)
    data = cleaning_fuel_consumption_var(data)
    data = cleaning_target_variables(data)
    data = encode_categorical_data(data)

    # Subsetting data to prepare for train/test split
    data = data[['Temperature','RPM','Fuel consumption','model_encoded','usage_encoded','membership_encoded','factory_china_encoded','factory_us_encoded','y']]

    # Subsetting data into X (independent variables) & y (dependent variable)
    X, y = data.iloc[:,:-1], data.iloc[:,-1]

    return X, y

if __name__== '__main__':
    main()