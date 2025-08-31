from evaluation import evaluate_model
from model import create_model, save_model
import pandas as pd

from training import train_model

if __name__ == '__main__':
    model = create_model()
    print("Model Created")

    data_path = "Cleaned_delinquency_prediction_dataset.csv"
    trained_model, X_test, y_test = train_model(data_path)

    save_model(trained_model)

    metrics = evaluate_model(trained_model, X_test, y_test)

    print(metrics)
