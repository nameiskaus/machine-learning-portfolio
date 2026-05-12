# qa_chain.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from chatbot import HFSeq2SeqPipeline

def load_vectorstore(index_path: str = "faiss_index"):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

    # 🔍 Check number of documents
    print(f"📚 Loaded vectorstore with {len(db.index_to_docstore_id)} documents.")

    return db

def retrieve_context(db, query: str, k: int = 3):
    return db.similarity_search(query, k=k)

def build_prompt(context_docs: list[Document], query: str):
    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""Answer the question based on the context below. Write a full paragraph.

                ### Context:
                {context}

                ### Question:
                {query}

                ### Answer:
                """
    return prompt


def load_qa_chain():
    # Load retrieval database
    db = load_vectorstore()
    
    # Load generation model
    generator = HFSeq2SeqPipeline(model_name="google/flan-t5-large")

    def qa_chain(question: str):
        context_docs = retrieve_context(db, question)

        # 🔍 DEBUG: Show what was retrieved
        print("\n--- Retrieved Documents ---")
        for i, doc in enumerate(context_docs):
            print(f"[Doc {i+1}]\n{doc.page_content}\n")

        prompt = build_prompt(context_docs, question)
        
        # 🔍 DEBUG: Show final prompt
        print("\n--- Final Prompt to Model ---")
        print(prompt)
        
        answer = generator(prompt)
        return answer


    return qa_chain