from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_texts(["init"], embeddings)

def store_docs(docs):
    vector_db.add_texts(docs)

def retrieve_docs(query):
    return vector_db.similarity_search(query, k=3)