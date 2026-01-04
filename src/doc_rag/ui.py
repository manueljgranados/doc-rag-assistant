import streamlit as st
import requests

st.set_page_config(page_title="Doc RAG Assistant", layout="wide")
st.title("Doc RAG Assistant (MVP)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Estado del backend")
    if st.button("Comprobar /health"):
        try:
            r = requests.get("http://localhost:8000/health", timeout=3)
            st.json(r.json())
        except Exception as e:
            st.error(f"No se pudo conectar: {e}")

with col2:
    st.subheader("Próximo paso")
    st.write("Aquí irán: subida de documentos, chat y panel de citas.")
