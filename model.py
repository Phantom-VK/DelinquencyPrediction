from sklearn.tree import DecisionTreeClassifier
import pickle

def create_model():
    """
    Create a Decision Tree Classifier model designed to predict
    customer delinquency based on financial features.
    :returns: Decision Tree Classifier model
    """
    model = DecisionTreeClassifier(random_state=42)
    return model

def save_model(model):
    filename = "model.pkl"
    with open(filename, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")
