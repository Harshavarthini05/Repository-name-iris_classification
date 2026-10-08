import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model_path = os.path.join("models", "best_iris_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model_and_scaler()

# Title and description
st.title("🌸 Iris Species Classifier")
st.markdown("""
Enter the flower measurements (in cm) to predict the iris species.  
This app uses a machine-learning model trained on the classic Iris dataset.
""")

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    - **Task:** Multi-class classification  
    - **Features:** Sepal length, sepal width, petal length, petal width  
    - **Classes:** Iris-setosa, Iris-versicolor, Iris-virginica  
    """)
    st.info("Used for internship project: IRIS CLASSIFICATION")

# Input form
with st.form("iris_form"):
    sepal_length = st.number_input("Sepal Length (cm)", value=5.1, step=0.1, min_value=0.0)
    sepal_width  = st.number_input("Sepal Width (cm)", value=3.5, step=0.1, min_value=0.0)
    petal_length = st.number_input("Petal Length (cm)", value=1.4, step=0.1, min_value=0.0)
    petal_width  = st.number_input("Petal Width (cm)", value=0.2, step=0.1, min_value=0.0)
    
    submitted = st.form_submit_button("Predict Species")

if submitted:
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    
    st.success(f"**Predicted Species:** {pred}")
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
        st.write("**Class probabilities:**")
        for cls, p in zip(model.classes_, probs):
            st.write(f"- **{cls}**: {p:.3f}")

# Optional: small footer
st.markdown("---")
st.caption("Internship Project: IRIS Classification | Model: scikit-learn")