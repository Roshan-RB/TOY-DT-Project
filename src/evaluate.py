import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_prep import load_data, preprocess_data

def evaluate_model():
    """Load the trained model and evaluate it on test data."""
    
    # Load data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Load the trained model
    model = joblib.load(r"C:\Users\rbhaskar\Desktop\decision_tree_project\models\best_decision_tree.pkl")

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", class_report)
    print("\nConfusion Matrix:\n", conf_matrix)

if __name__ == "__main__":
    evaluate_model()
