import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, \
    average_precision_score, matthews_corrcoef, cohen_kappa_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

selections = ("correlation", "gain-ratio", "info-gain", "intuition", "wrapper-subset")

models = {

    "Decision Tree": DecisionTreeClassifier(max_depth = 3),
    "QDA": QuadraticDiscriminantAnalysis(),
    "Logistic Regression": LogisticRegression(max_iter = 1000),
    "SVC": SVC(probability = True)

}

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

def results(y_test, prediction):

    cm = confusion_matrix(y_test, prediction)
    correct = np.trace(cm)
    incorrect = np.sum(cm) - correct

    accuracy = accuracy_score(y_test, prediction)
    kappa = cohen_kappa_score(y_test, prediction)
    mae = mean_absolute_error(y_test, prediction)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))

    rae = mae / np.mean(np.abs(y_test - np.mean(y_test)))
    rrse = rmse / np.sqrt(np.mean((y_test - np.mean(y_test)) ** 2))

    instances = len(y_test)

    print(f"Accuracy: {100 * accuracy:.4f}%")

    print(f"Correctly Classified Instances: {correct}")
    print(f"Incorrectly Classified Instances: {incorrect}")
    print(f"Kappa Statistic: {kappa:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Relative Absolute Error (RAE): {rae:.4f}")
    print(f"Root Relative Squared Error (RRSE): {rrse:.4f}")
    print(f"Total Number of Instances: {instances}")

def metrics(x_test, y_test, model, prediction):

    report = classification_report(y_test, prediction, output_dict = True)
    cm = confusion_matrix(y_test, prediction)

    tp_rate = np.diag(cm) / np.sum(cm, axis = 1)
    fp_rate = 1 - np.diag(cm) / np.sum(cm, axis = 0)

    mcc = matthews_corrcoef(y_test, prediction)
    probas = model.predict_proba(x_test)[:, 1]
    roc_area = roc_auc_score(y_test, probas)
    prc_area = average_precision_score(y_test, probas)

    columns = ('TP Rate', 'FP Rate', 'Precision', 'Recall', 'F-Measure', 'MCC', 'ROC Area', 'PRC Area')
    rows = []

    for label, metrics in report.items():

        if label not in ('accuracy', 'macro avg', 'weighted avg'):

            rows.append([tp_rate[int(label)], fp_rate[int(label)], metrics['precision'], metrics['recall'], metrics['f1-score'], mcc, roc_area, prc_area])

    weighted_avg = report['weighted avg']
    rows.append([np.mean(tp_rate), np.mean(fp_rate), weighted_avg['precision'], weighted_avg['recall'], weighted_avg['f1-score'], mcc, roc_area, prc_area])

    data = pd.DataFrame(rows, columns = columns)
    data.index = list(report.keys())[:-3] + ['Weighted Avg']

    return data, cm

def main():

    for selection in selections:

        print(f"\nFeature Selection: {selection}")
        print("=" * 100)

        x_train, x_val, x_test, y_train, y_val, y_test = load(selection)

        for name, model in models.items():

            print(f"\nClassifier: {name}")
            print("-" * 100)

            model.fit(x_train, y_train)

            eval = model.predict(x_val)
            accuracy = accuracy_score(y_val, eval)
            print(f"Validation Accuracy: {100 * accuracy:.4f}%")

            prediction = model.predict(x_test)
            results(y_test, prediction)

            data, cm = metrics(x_test, y_test, model, prediction)
            print(data.to_string())

            print("\nConfusion Matrix:")
            print(f"{'':<10}Predicted 0   Predicted 1")
            print(f"Actual 0  {cm[0, 0]:>10}   {cm[0, 1]:>10}")
            print(f"Actual 1  {cm[1, 0]:>10}   {cm[1, 1]:>10}")


if __name__ == "__main__":

    main()

# Anieesh Saravanan, 6, 2025