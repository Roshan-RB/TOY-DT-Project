import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath=r"C:\Users\rbhaskar\Desktop\decision_tree_project\data\iris.csv"):
    """Load dataset from CSV file"""
    df = pd.read_csv(filepath)

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("Warning: Dataset contains missing values!")
        print(df.isnull().sum())

    # Map target values
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    return df

def preprocess_data(df):
    """Split dataset into train/test sets and scale features"""
    X = df.drop(columns=['species'])
    y = df['species']

    # Check again for missing values
    if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
        print("Error: NaN values found in dataset!")
        print(X.isnull().sum())
        print(y.isnull().sum())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    print("Data loaded and preprocessed successfully!")
