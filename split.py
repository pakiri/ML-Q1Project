import os

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

attributes_to_remove = [

    # add list of attributes as strings to remove here

]

folder_name = ""

def preview(data, title = "Preview"):

    print(f"{title}:\n{data.head(3).to_string()}\n")

data = pd.read_csv('Q1 Project/datasets/mushrooms.csv')
print(f"{data.shape[0]} instances")

data.replace('?', np.nan, inplace = True)

data['class'] = data['class'].map({'e': 0, 'p': 1})

x = data.drop(columns='class')
y = data['class']

x_encoded = pd.DataFrame()
mappings = {}

for column in x.columns:

    x[column] = x[column].astype('category')
    x_encoded[column] = x[column].cat.codes
    mappings[column] = dict(enumerate(x[column].cat.categories))
    x_encoded[column].replace(-1, np.nan, inplace = True)

knn_imputer = KNNImputer(n_neighbors = 5)
x_imputed = knn_imputer.fit_transform(x_encoded)
x = pd.DataFrame(x_imputed, columns=x.columns)

label_encoders = {}

for column in x.columns:

    le = LabelEncoder()
    x[column] = le.fit_transform(x[column].round().astype(int))
    label_encoders[column] = le

for column in x.columns:

    x[column] = x[column].round().astype(int).map(mappings[column])

def remove_attributes_and_split(x, y, attributes, name_suffix):

    x_temp = x.drop(columns = attributes)

    x_train, x_temp_split, y_train, y_temp_split = train_test_split(

        x_temp, y, test_size = 0.2, random_state = 42, stratify = y
    )

    x_val, x_test, y_val, y_test = train_test_split(

        x_temp_split, y_temp_split, test_size = 0.5, random_state = 42, stratify = y_temp_split

    )

    if not os.path.exists(f'Q1 Project/datasets/{name_suffix}'):

        os.makedirs(f'Q1 Project/datasets/{name_suffix}')

    x_train.to_csv(f'Q1 Project/datasets/{name_suffix}/x_train.csv', index = False)
    x_val.to_csv(f'Q1 Project/datasets/{name_suffix}/x_val.csv', index = False)
    x_test.to_csv(f'Q1 Project/datasets/{name_suffix}/x_test.csv', index = False)

    y_train.to_csv(f'Q1 Project/datasets/{name_suffix}/y_train.csv', index = False)
    y_val.to_csv(f'Q1 Project/datasets/{name_suffix}/y_val.csv', index = False)
    y_test.to_csv(f'Q1 Project/datasets/{name_suffix}/y_test.csv', index = False)

    preview(x_train, f"Training Set - {name_suffix}")

remove_attributes_and_split(x, y, attributes_to_remove, folder_name)

# Anieesh Saravanan, 6, 2025