import joblib
from pathlib import Path

# Load the model
model_path = Path(__file__).resolve().parent / "iris.joblib"
model = joblib.load(model_path)

# Define a predict function
def predict(features):
    """
    Predict the Iris species given input features.
    :param features: list or array-like of 4 float values [sepal_length, sepal_width, petal_length, petal_width]
    :return: string with predicted species name
    """
    prediction = model.predict([features])
    # Map numeric class to Iris species
    iris_classes = ['setosa', 'versicolor', 'virginica']
    return iris_classes[prediction[0]]
