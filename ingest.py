# ingest.py
import os, glob
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_DIR = "data"
DB_DIR = "chroma_db"
EMB_MODEL = "BAAI/bge-m3"  # ou "all-MiniLM-L6-v2" si tu veux plus léger

def load_docs():
    docs = []
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    print(f"📂 {len(pdf_files)} PDF trouvés dans {DATA_DIR}")
    for f in pdf_files:
        loader = PyPDFLoader(f)
        pages = loader.load()
        print(f"📄 {len(pages)} pages dans {f}")
        for page in pages:
            docs.append({"text": page.page_content,
                         "meta": {"source": f, "page": page.metadata.get("page", None)}})
    print(f"✅ Total documents chargés : {len(docs)}")
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = []
    for d in docs:
        splitted = splitter.split_text(d["text"])
        for c in splitted:
            chunks.append({"text": c, "meta": d["meta"]})
    print(f"✂️ Total chunks créés : {len(chunks)}")
    return chunks

def build_index(chunks):
    if len(chunks) == 0:
        print("⚠️ Aucun chunk à indexer, abandon de la construction de l'index.")
        return

    client = chromadb.PersistentClient(path=DB_DIR)
    coll = client.get_or_create_collection("corpus")
    print("⚡ Encodage des textes en embeddings…")
    model = SentenceTransformer(EMB_MODEL)
    texts = [c["text"] for c in chunks]
    metas = [c["meta"] for c in chunks]
    ids   = [f"id_{i}" for i in range(len(chunks))]
    embs  = model.encode(texts, normalize_embeddings=True)
    coll.add(ids=ids, embeddings=embs.tolist(), documents=texts, metadatas=metas)
    print(f"✅ {len(chunks)} chunks indexés dans {DB_DIR}")

if __name__ == "__main__":
    print("🚀 Début de l'ingestion")
    docs = load_docs()
    chunks = split_docs(docs)
    build_index(chunks)
    print("🏁 Ingestion terminée")
