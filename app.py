import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import requests
import json

# ------------------------
# Config
# ------------------------
INDEX_DIR = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 300
MODEL_NAME = "llama3:8b"

# ------------------------
# Chargement embeddings et index FAISS
# ------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # plus de documents pour enrichir la réponse

# ------------------------
# Wrapper Ollama HTTP (réponse complète)
# ------------------------
def generate_complete(prompt: str, temperature=0.0, max_tokens=MAX_TOKENS):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        r = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
        r.raise_for_status()
        full_response = ""
        for line in r.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    text = data.get("response", "")
                    if text:
                        full_response += text
                except json.JSONDecodeError:
                    continue
        return full_response
    except requests.exceptions.RequestException as e:
        return f"Error contacting Ollama API: {e}"

# ------------------------
# Prompt template amélioré
# ------------------------
TEMPLATE = """
You are a domain expert in epidemiology and machine learning.
Answer the QUESTION using **only** the provided CONTEXT.
Provide a **detailed and structured explanation**, including:
- Introduction to the concept
- Technical description and mechanisms
- Formulas or pseudo-code if relevant
- Examples and applications
- Any limitations mentioned in the context

If the information is missing in the context, say explicitly that you don't know.

QUESTION: {question}

CONTEXT:
{context}
"""

prompt_template = PromptTemplate(template=TEMPLATE, input_variables=["question", "context"])

# ------------------------
# Streamlit UI
# ------------------------
st.title("🦠 Pandemic RAG - Q&A (CPU, Detailed Responses)")
st.write("Posez une question sur la pandémie ou la modélisation GNN/LSTM.")

question = st.text_input("Votre question")
if not question:
    st.info("Entrez une question pour commencer.")
    st.stop()

# ------------------------
# Récupération contexte FAISS
# ------------------------
docs = retriever.get_relevant_documents(question)
context_text = "\n".join([doc.page_content for doc in docs])
prompt = prompt_template.format(question=question, context=context_text)

# ------------------------
# Génération de la réponse complète
# ------------------------
st.subheader("Réponse")
with st.spinner("Génération de la réponse…"):
    answer = generate_complete(prompt)
st.write(answer)

# ------------------------
# Affichage sources
# ------------------------
st.subheader("Sources")
seen = set()
for doc in docs:
    src = doc.metadata.get("source", "inconnu")
    if src not in seen:
        st.write(f"- {src}")
        seen.add(src)
