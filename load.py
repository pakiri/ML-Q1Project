import pickle

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

model_path = "Q1 Project/models/svc_correlation.pkl"
selection = "correlation"

def load(selection):

    x_train = pd.read_csv(f'Q1 Project/classification/{selection}/x_train.csv')
    x_val = pd.read_csv(f'Q1 Project/classification/{selection}/x_val.csv')
    x_test = pd.read_csv(f'Q1 Project/classification/{selection}/x_test.csv')
    y_train = pd.read_csv(f'Q1 Project/classification/{selection}/y_train.csv').values.ravel()
    y_val = pd.read_csv(f'Q1 Project/classification/{selection}/y_val.csv').values.ravel()
    y_test = pd.read_csv(f'Q1 Project/classification/{selection}/y_test.csv').values.ravel()

    encoder = LabelEncoder()

    for col in x_train.columns:

        encoder.fit(x_train[col])

        for dataset in (x_train, x_val, x_test):

            dataset[col] = encoder.transform(dataset[col])

    return x_train, x_val, x_test, y_train, y_val, y_test

def load_model(path):

    with open(path, "rb") as file:

        model = pickle.load(file)

    return model

x_train, x_val, x_test, y_train, y_val, y_test = load(selection)

model = load_model(model_path)
prediction = model.predict(x_test)

print(f"Prediction Accuracy: {accuracy_score(y_test, prediction) * 100:.4f}%")

# Anieesh Saravanan, 6, 2025