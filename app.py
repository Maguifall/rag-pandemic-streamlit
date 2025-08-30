import streamlit as st
import traceback

# Bloc pour afficher les erreurs dans Streamlit
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except Exception as e:
    st.error("Erreur au lancement de l'app :")
    st.text(traceback.format_exc())
    st.stop()

st.set_page_config(page_title="RAG Medical QA", page_icon="🩺")
st.title("🩺 RAG Medical QA Assistant - Version légère")

# Constantes pour test
DB_DIR = "chroma_db"
EMB_MODEL = "all-MiniLM-L6-v2"  # modèle très léger pour test

@st.cache_resource
def get_resources():
    client = chromadb.PersistentClient(path=DB_DIR)
    coll = client.get_or_create_collection("corpus")
    emb = SentenceTransformer(EMB_MODEL)
    return coll, emb

coll, emb = get_resources()

def retrieve(q, top_k=5):
    q_vec = emb.encode([q], normalize_embeddings=True).tolist()
    res = coll.query(query_embeddings=q_vec, n_results=top_k, include=["documents","metadatas"])
    ctx_items = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        ctx_items.append((doc, meta))
    return ctx_items

user_q = st.text_input("Pose ta question médicale :")
if user_q:
    ctx = retrieve(user_q, top_k=5)
    st.markdown("### 📄 Documents récupérés")
    for i, (doc, meta) in enumerate(ctx, 1):
        st.markdown(f"**[{i}]** {doc}\n(Source: {meta.get('source')}, page={meta.get('page')})")

    # Réponse factice pour test frontend
    st.markdown("### 📝 Réponse (test)")
    st.write("Voici un exemple de réponse générée pour vérifier le frontend.")
