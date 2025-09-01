import os
from pathlib import Path
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS

# -----------------------------
# Config
# -----------------------------
PDF_DIR = Path("data")
FAISS_INDEX_PATH = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # léger & fiable
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# -----------------------------
# Vérifier l'existence du dossier PDFs
# -----------------------------
if not PDF_DIR.exists():
    raise FileNotFoundError(f"Le dossier {PDF_DIR} n'existe pas. Créez-le et ajoutez vos PDFs.")

# Charger tous les PDFs
pdf_files = list(PDF_DIR.glob("*.pdf"))
print(f"{len(pdf_files)} fichiers PDF trouvés.")

if not pdf_files:
    raise FileNotFoundError("Aucun PDF trouvé dans ./data. Ajoutez des documents et réessayez.")

# -----------------------------
# Extraire les documents
# -----------------------------
documents = []
for pdf in pdf_files:
    loader = PyPDFLoader(str(pdf))
    docs = loader.load()
    for d in docs:
        # Ajouter la source comme metadata
        d.metadata = d.metadata or {}
        d.metadata["source"] = pdf.name
    documents.extend(docs)

print(f"{len(documents)} pages extraites.")

# -----------------------------
# Split into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)

splits = text_splitter.split_documents(documents)
print(f"{len(splits)} chunks générés.")

# -----------------------------
# Embeddings HuggingFace
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# -----------------------------
# Créer et sauvegarder l'index FAISS
# -----------------------------
print("Création de l'index FAISS…")
vectorstore = FAISS.from_documents(splits, embeddings)
vectorstore.save_local(FAISS_INDEX_PATH)
print("Index FAISS sauvegardé ✅")
