import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import time

# Load the trained model and scaler
model = joblib.load('voting_regressor_no_log.pkl')
scaler = joblib.load('scaler_no_log.pkl')

# Custom CSS for modern and sleek design
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background: linear-gradient(to bottom right, #f4f7f6, #dbe6e4);
        color: #333;
        margin: 0;
    }
    .main-title {
        font-size: 40px;
        color: #2c3e50;
        text-align: center;
        font-weight: bold;
        margin-bottom: 40px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .card {
        background-color: #ffffff;
        padding: 20px;
        margin: 15px 0;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .card-header {
        font-size: 24px;
        font-weight: 600;
        color: #34495e;
    }
    .prediction {
        font-size: 32px;
        color: #27ae60;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    .recommendation {
        font-size: 18px;
        color: #e74c3c;
        background-color: #f9c2c2;
        padding: 10px;
        border-radius: 5px;
        margin-top: 15px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #7f8c8d;
        margin-top: 50px;
        font-style: italic;
    }
    .chart-container {
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("""
    <div class="main-title">✨ Prediksi Biaya Medis Anda ✨</div>
""", unsafe_allow_html=True)

# Main Section: Input + Prediction
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">🔍 Masukkan Data Anda</div>', unsafe_allow_html=True)

    # Layout Grid: 2 columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Usia (Tahun):", min_value=0, max_value=120, value=30, step=1)
        bmi = st.number_input("BMI (Indeks Massa Tubuh):", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
        children = st.number_input("Jumlah Anak/Tanggungan:", min_value=0, max_value=10, value=0, step=1)

    with col2:
        sex = st.selectbox("Jenis Kelamin:", options=["Laki-laki", "Perempuan"])
        smoker = st.selectbox("Status Merokok:", options=["Ya", "Tidak"])
        region = st.selectbox("Wilayah:", options=["Northeast", "Northwest", "Southeast", "Southwest"])

    # Convert categorical inputs
    sex = 1 if sex == "Laki-laki" else 0
    smoker = 1 if smoker == "Ya" else 0
    region_mapping = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}
    region = region_mapping[region]

    # Create DataFrame for input data
    input_data = pd.DataFrame({
        'age': [age],
        'children': [children],
        'smoker': [smoker],
        'region': [region],
        'bmi': [bmi],
        'sex': [sex]
    })

    st.markdown('</div>', unsafe_allow_html=True)

    # Adding a Progress Bar for Model Prediction
    with st.spinner('Memproses data dan menghasilkan prediksi...'):
        time.sleep(2)  # Simulate model processing time
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📊 Hasil Prediksi</div>', unsafe_allow_html=True)

    # Display prediction
    st.markdown(f"""
        <div class="prediction">Prediksi biaya medis Anda adalah:<br>
        <span style='color: #27ae60;'>${prediction:,.2f}</span></div>
    """, unsafe_allow_html=True)

    # Recommendations based on conditions
    if smoker == 1:
        st.markdown("<div class='recommendation'>🚭 Pertimbangkan untuk berhenti merokok untuk mengurangi risiko kesehatan.</div>", unsafe_allow_html=True)
    if bmi > 30:
        st.markdown("<div class='recommendation'>⚠️ BMI Anda menunjukkan obesitas. Pertimbangkan program diet atau olahraga.</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Visualizations Section
with st.container():
    st.markdown('<div class="card chart-container">', unsafe_allow_html=True)

    # Create an interactive plot using Plotly
    st.markdown("<div class='card-header'>📊 Visualisasi Data</div>", unsafe_allow_html=True)

    # Interactive Bar Chart with Plotly
    fig = px.bar(
        x=['Usia', 'Jumlah Anak', 'BMI', 'Status Merokok', 'Wilayah', 'Jenis Kelamin'],
        y=[age, children, bmi, smoker, region, sex],
        labels={'x': 'Fitur', 'y': 'Nilai'},
        title="Input Data vs Prediksi Biaya Medis"
    )
    st.plotly_chart(fig)

    # Display Distribution Chart for Medical Costs (Simulated)
    st.markdown("<div class='card-header'>📈 Distribusi Biaya Medis (Simulasi)</div>", unsafe_allow_html=True)

    # Example: Simulated distribution of medical costs for visualization
    np.random.seed(42)
    simulated_data = np.random.normal(loc=5000, scale=2000, size=1000)  # Example distribution

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(simulated_data, kde=True, color='orange', bins=30)
    ax.set_title('Distribusi Biaya Medis (Simulasi)')
    ax.set_xlabel('Biaya Medis')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        Dibuat dengan ❤️ oleh Tim Prediksi Medis • © 2025
    </div>
""", unsafe_allow_html=True)
