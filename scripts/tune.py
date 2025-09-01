import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler



def tune_hyperparameters(data_path):
    """
    Tune model hyperparameters
    :param data_path: Path to dataset
    :return: best estimator, X_test, y_test
    """
    df = pd.read_csv(data_path)

    X = df.drop(columns=['Delinquent_Account'])
    y = df['Delinquent_Account']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    print("Applying Standard Scaler")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    model = DecisionTreeClassifier(random_state=42)

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15, 20, 30],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': [None, 'sqrt', 'log2', 0.8, 1.0],
        'splitter': ['best', 'random']
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv = 5,
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )
    print("Tuning hyperparameters")

    grid_search.fit(X_train_scaled, y_train)
    print("Best score: {:.3f}".format(grid_search.best_score_))
    print("Best params: {}".format(grid_search.best_params_))
    return grid_search.best_estimator_, X_test_scaled, y_test