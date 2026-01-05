import streamlit as st
import requests

API = "http://localhost:8000"

st.set_page_config(page_title="Doc RAG Assistant", layout="wide")
st.title("Doc RAG Assistant")

with st.sidebar:
    st.header("Índice global")
    uploaded = st.file_uploader("Subir PDF/MD (máx. 20 MB)", type=["pdf", "md", "markdown"])
    if uploaded and st.button("1) Subir"):
        files = {"file": (uploaded.name, uploaded.getvalue())}
        r = requests.post(f"{API}/documents/upload", files=files, timeout=60)
        st.write(r.json())

    if st.button("2) Reindexar todo (global)"):
        r = requests.post(f"{API}/documents/reindex", timeout=600)
        st.write(r.json())

    st.divider()
    use_openai = st.checkbox("Usar OpenAI (si hay API key)", value=False)
    use_rerank = st.checkbox("Usar reranker (mejor precisión)", value=True)
    top_k = st.slider("Top-K", 1, 10, 5)

question = st.text_input("Pregunta")
if st.button("Consultar") and question.strip():
    payload = {"question": question, "top_k": top_k, "use_openai": use_openai, "use_rerank": use_rerank}
    r = requests.post(f"{API}/query", json=payload, timeout=120)
    if r.status_code != 200:
        st.error(r.text)
    else:
        data = r.json()
        st.subheader("Respuesta")
        st.write(data["answer"])

        st.subheader("Citas")
        for c in data["citations"]:
            title = f"{c['source_filename']} | {c['anchor']} | score={c['score']:.3f}"
            with st.expander(title):
                st.write(c["snippet"])
