import streamlit as st
import requests

st.title("Chat PDFs")

pergunta = st.text_input("Digite sua pergunta:")

if st.button("Perguntar"):
    resposta = requests.post("http://127.0.0.1:5000/chat", json={"pergunta": pergunta}).json()
    st.write("Resposta:", resposta["resposta"])
