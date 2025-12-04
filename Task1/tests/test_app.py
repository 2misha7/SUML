from app.predict import predict


def test_predict_returns_valid_class():
    """Ensure predict returns one of the expected Iris classes for a sample input."""
    features = [5.1, 3.5, 1.4, 0.2]
    result = predict(features)
    valid_classes = {"setosa", "versicolor", "virginica", 0, 1, 2}
    assert result in valid_classes
