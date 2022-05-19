import streamlit as st

st.write("# Classificação de Cancer")
st.write("## Breast Cancer Wisconsin (Diagnostic)")

st.sidebar.write("### Parâmetros")
st.sidebar.slider("Perimeter", 49.0, 251.0, 107.0, 0.1)
st.sidebar.slider("Area", 184.0, 4254.0, 881.0, 0.1)
st.sidebar.slider("Compactness", 0.01, 1.6, 0.25, 0.1)
st.sidebar.slider("Concavity", 0.0, 1.30, 0.27, 0.1)
