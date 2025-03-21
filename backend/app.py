import os
import pdfplumber
import faiss
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Carregar modelo de embeddings
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# Caminho dos PDFs
DATA_DIR = "data"

# Criar índice FAISS
dimensao = 384  # Tamanho do embedding do modelo
index = faiss.IndexFlatL2(dimensao)

# Função para extrair texto dos PDFs
def extrair_texto_pdfs():
    textos = []
    for arquivo in os.listdir(DATA_DIR):
        if arquivo.endswith(".pdf"):
            caminho = os.path.join(DATA_DIR, arquivo)
            with pdfplumber.open(caminho) as pdf:
                texto = " ".join([pagina.extract_text() for pagina in pdf.pages if pagina.extract_text()])
                textos.append(texto)
    return textos

# Criar base vetorizada
textos = extrair_texto_pdfs()
vetores = modelo.encode(textos)
index.add(np.array(vetores, dtype=np.float32).reshape(-1, dimensao))

@app.route('/chat', methods=['POST'])
def chat():
    pergunta = request.json.get("pergunta", "")
    pergunta_vetor = modelo.encode([pergunta])
    _, indices = index.search(np.array(pergunta_vetor), k=1)  # Busca o texto mais próximo
    resposta = textos[indices[0][0]] if textos else "Nenhuma informação encontrada."
    
    return jsonify({"resposta": resposta})

if __name__ == '__main__':
    app.run(debug=True)
