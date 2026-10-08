# Iris Classification

A machine learning project that classifies Iris flowers into three species using their sepal and petal measurements.

## Project Overview

This project demonstrates an end-to-end machine learning workflow, including:

* Data loading and preprocessing
* Exploratory Data Analysis (EDA)
* Feature scaling
* Model training
* Model evaluation
* Best model selection and saving
* Example inference using the trained model

## Dataset

The project uses the Iris dataset containing measurements of:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The target classes are:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

## Project Structure

```text
iris_classification/
├── data/
│   └── iris.csv
├── models/
│   ├── best_iris_model.pkl
│   └── scaler.pkl
├── iris_classification.ipynb
├── inference_example.py
└── README.md
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook
* Joblib

## Machine Learning Workflow

1. Load the Iris dataset
2. Perform data preprocessing
3. Explore the dataset using EDA
4. Split the data into training and testing sets
5. Apply feature scaling
6. Train multiple classification models
7. Compare model performance
8. Select the best-performing model
9. Save the trained model and scaler
10. Perform predictions using the inference script

## Model Files

`best_iris_model.pkl` contains the selected trained machine learning model.

`scaler.pkl` contains the feature scaler used during model training.

Both are required for performing inference on new data.

## Running the Project

### Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter joblib
```

### Run the Notebook

```bash
jupyter notebook
```

Open:

```text
iris_classification.ipynb
```

### Run the Inference Example

```bash
python inference_example.py
```

## Example Input

```text
Sepal Length: 5.1
Sepal Width: 3.5
Petal Length: 1.4
Petal Width: 0.2
```

The model will predict the corresponding Iris species.

## Purpose

This project was developed to demonstrate practical understanding of machine learning classification, data preprocessing, model evaluation, and deployment-ready inference.

## Author

**Harshavarthini K**

Mechanical & Automobile Engineer
Interested in Automotive Systems, Vehicle Testing and Software Testing.
