import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import create_model

def train_model(data_path):
    """
    SPlit data into train and test sets
    :param data_path: path to data file
    :return: trained model, X_test, y_test
    """
    df = pd.read_csv(data_path)

    X = df.drop(columns=['Delinquent_Account'])
    y = df['Delinquent_Account']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_model()

    model.fit(X_train, y_train)

    return model, X_test, y_test
