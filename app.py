# app.py
import streamlit as st
import traceback

# Bloc pour afficher les erreurs d'import directement dans Streamlit
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    from langchain_community.chat_models import ChatOllama
    from langchain.prompts import PromptTemplate
except Exception as e:
    st.error("Erreur au lancement de l'app :")
    st.text(traceback.format_exc())
    st.stop()

# Configuration de la page
st.set_page_config(page_title="RAG Medical QA", page_icon="🩺")
st.title("🩺 RAG Medical QA Assistant")

# Constantes
DB_DIR = "chroma_db"
EMB_MODEL = "BAAI/bge-m3"
LLM_MODEL = "llama3.1:8b-instruct"  # Ou mistral:7b-instruct

# Chargement des ressources
@st.cache_resource
def get_resources():
    client = chromadb.PersistentClient(path=DB_DIR)
    coll = client.get_or_create_collection("corpus")
    emb  = SentenceTransformer(EMB_MODEL)
    llm  = ChatOllama(model=LLM_MODEL, temperature=0.1)
    return coll, emb, llm

coll, emb, llm = get_resources()

# Fonction pour récupérer les documents pertinents
def retrieve(q, top_k=5):
    q_vec = emb.encode([q], normalize_embeddings=True).tolist()
    res = coll.query(query_embeddings=q_vec, n_results=top_k, include=["documents","metadatas"])
    ctx_items = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        ctx_items.append((doc, meta))
    return ctx_items

# Prompt pour le modèle
PROMPT = PromptTemplate.from_template(
    "You are a research assistant specialized in pandemics. Use ONLY the context below to answer.\n"
    "If you cannot find the answer, say 'I don't know based on the provided documents.'\n"
    "Always cite sources (file, page).\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

# Interface utilisateur
user_q = st.text_input("Pose ta question médicale :")
if user_q:
    ctx = retrieve(user_q, top_k=5)
    context_text = "\n\n".join(
        [f"[{i+1}] {c[0]}\n(Source: {c[1].get('source')}, page={c[1].get('page')})"
         for i, c in enumerate(ctx)]
    )
    prompt = PROMPT.format(context=context_text, question=user_q)
    with st.spinner("Réflexion du modèle..."):
        answer = llm.invoke(prompt)
    st.markdown("### 📝 Réponse")
    st.write(answer)

    with st.expander("📚 Sources"):
        for i, (doc, meta) in enumerate(ctx, 1):
            st.markdown(f"**[{i}]** {meta.get('source')} — page {meta.get('page')}")
    st.caption("⚠️ Les réponses générées s'appuient sur les articles et rapports fournis. "
               "Elles doivent être utilisées uniquement dans un cadre académique et ne constituent pas une conclusion définitive.")
