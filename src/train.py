import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from data_prep import load_data, preprocess_data

def tune_decision_tree():
    """Perform hyperparameter tuning on Decision Tree."""

    # Load data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Define parameter grid
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 4, 5, 6, 7, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    # Initialize Decision Tree and GridSearchCV
    model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, r"C:\Users\rbhaskar\Desktop\decision_tree_project\models\best_decision_tree.pkl")

    print("Best model parameters:", grid_search.best_params_)
    print("Best model saved successfully!")

if __name__ == "__main__":
    tune_decision_tree()
