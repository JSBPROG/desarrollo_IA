import streamlit as st
import requests

st.title("🧮 Predicción de Fuga Bancaria")

# Inputs del usuario
credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=600)
geography = st.selectbox("Geografía", ["France", "Spain", "Germany"])
gender = st.selectbox("Género", ["Male", "Female"])
age = st.number_input("Edad", min_value=18, max_value=100, value=35)
tenure = st.slider("Tenencia (años)", 0, 10, 3)
balance = st.number_input("Balance", min_value=0.0, step=100.0, value=50000.0)
num_of_products = st.selectbox("Número de productos", [1, 2, 3, 4])
has_cr_card = st.selectbox("¿Tiene tarjeta de crédito?", [0, 1])
is_active_member = st.selectbox("¿Es miembro activo?", [0, 1])
estimated_salary = st.number_input("Salario estimado", min_value=0.0, step=1000.0, value=100000.0)
complain = st.selectbox("¿Se ha quejado?", [0, 1])
satisfaction_score = st.slider("Puntaje de satisfacción", 1, 5, 3)
card_type = st.selectbox("Tipo de tarjeta", ["DIAMOND", "GOLD", "PLATINUM", "SILVER"])

# Enviar datos a la API
if st.button("Predecir"):
    cliente_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Complain": complain,
        "Satisfaction Score": satisfaction_score,
        "Card Type": card_type
    }

    try:
        response = requests.post("http://localhost:5000/prediccion", json=cliente_data)

        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Probabilidad de abandono: {result['probabilidad']:.2f}")
            st.info(f"🔍 Predicción: {'Abandona' if result['abandono'] else 'Permanece'}")
        else:
            st.error(f"Error en la predicción: {response.text}")
    except Exception as e:
        st.error(f"Error de conexión con la API: {str(e)}")



