# ChatPDF - Inteligência Artificial para Análise de Documentos PDF

## Visão Geral

Este projeto tem como objetivo criar um chat interativo capaz de responder com base no conteúdo de arquivos PDF fornecidos pelo usuário. Utilizando conceitos de IA generativa, embeddings e buscas vetorizadas para estruturar um sistema eficiente de recuperação de informações.

## Tecnologias Utilizadas
- **Python**
- **Azure OpenAI**
- **FAISS** (para buscas vetorizadas)
- **Streamlit** (para interface do chat)
- **PyMuPDF** (para extração de texto dos PDFs)

## Passo a Passo da Implementação

### 1. Configuração do Ambiente

```sh
py -m venv venv
venv\Scripts\activate
py -m pip install -r requirements.txt
```

### 2. Executando o Backend

```sh
py backend.py
```

### 3. Rodando o Frontend

```sh
streamlit run app.py
```

## Prints do Processo

### 🔹 Execução do Backend
![image](https://github.com/user-attachments/assets/d4564aea-14d5-4574-9c0a-7addd0e9796c)

### 🔹 Interface do Chat
![image](https://github.com/user-attachments/assets/9ff68088-9910-421a-91ee-1678848bb0cc)

## Insights e Possibilidades
- A IA conseguiu interpretar e resumir artigos científicos de maneira eficiente.

- A busca vetorial torna a recuperação de informações extremamente rápida e precisa.
- Possível integração com outras plataformas para expansão do projeto.

## Como Contribuir
1. Faça um fork deste repositório.
2. Clone para sua máquina.
3. Crie uma nova branch.
4. Implemente as melhorias.
5. Envie um pull request.

Sinta-se à vontade para sugerir melhorias! 🚀

