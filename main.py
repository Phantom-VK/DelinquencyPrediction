from scripts.evaluation import evaluate_model
from scripts.model import create_model, save_model

from scripts.tune import tune_hyperparameters

if __name__ == '__main__':
    model = create_model()
    print("Model Created")

    data_path = "Resources/Cleaned_delinquency_prediction_dataset.csv"
    # trained_model, X_test, y_test = train_model(data_path)
    best_model, X_test, y_test = tune_hyperparameters(data_path)
    save_model(best_model)

    metrics = evaluate_model(best_model, X_test, y_test)

    print(metrics)
