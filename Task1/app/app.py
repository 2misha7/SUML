import streamlit as st

from predict import predict

st.set_page_config(page_title="Iris Classifier Version 2")

st.title("Iris Flower Species Predictor")
st.write("Enter the flower's measurements below (in centimeters) to predict its species:")

# Typical Iris feature ranges:
# Sepal length: 4.3–7.9
# Sepal width: 2.0–4.4
# Petal length: 1.0–6.9
# Petal width: 0.1–2.5

sepal_length = st.number_input("Sepal length (cm)", min_value=4.0, max_value=8.0, step=0.1, help="Typical range: 4.3–7.9 cm")
sepal_width = st.number_input("Sepal width (cm)", min_value=2.0, max_value=4.5, step=0.1, help="Typical range: 2.0–4.4 cm")
petal_length = st.number_input("Petal length (cm)", min_value=1.0, max_value=7.0, step=0.1, help="Typical range: 1.0–6.9 cm")
petal_width = st.number_input("Petal width (cm)", min_value=0.1, max_value=2.5, step=0.1, help="Typical range: 0.1–2.5 cm")

if st.button("Predict"):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    result = predict(features)
    st.success(f"The predicted Iris species is: **{result.capitalize()}**")
else:
    st.info("Adjust the measurements and click **Predict** to see the result.")

