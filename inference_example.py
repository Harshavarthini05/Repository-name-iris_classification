# Cell: Inference example (no warnings)
import joblib
import numpy as np

model = joblib.load('models/best_iris_model.pkl')
scaler = joblib.load('models/scaler.pkl')

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return pred

# Example
sepal_length = 5.1
sepal_width = 3.5
petal_length = 1.4
petal_width = 0.2

species = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
print(f"Predicted species: {species}")