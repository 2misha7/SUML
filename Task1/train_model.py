from pathlib import Path

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import joblib

def train_model():
    # load the iris dataset
    iris = datasets.load_iris()
    x,y = iris.data, iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=5)
    # Train the model
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Save model to joblib file
    # Ensure app/ folder exists next to this script
    app_dir = Path(__file__).resolve().parent / "app"
    app_dir.mkdir(exist_ok=True)
    joblib.dump(clf, str(app_dir / 'iris.joblib'))

if __name__ == "__main__":
    train_model()