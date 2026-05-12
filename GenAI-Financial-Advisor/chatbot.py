# chatbot.py
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

class HFSeq2SeqPipeline:
    def __init__(self, model_name="google/flan-t5-large"):
        self.pipe = pipeline("text2text-generation", model=model_name)

    def __call__(self, prompt: str, max_new_tokens=1024):
        result = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            clean_up_tokenization_spaces=True,
        )
        return result[0]["generated_text"]

def load_vectorstore(index_path: str = "faiss_index"):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    return db

def retrieve_documents(db, query: str, k: int = 3):
    docs = db.similarity_search(query, k=k)
    return docs

def build_prompt(context_docs: list[Document], query: str):
    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = (
        f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    return prompt
