# GenAI Financial Advisor: Interview Cheat Sheet

## 1. The Elevator Pitch (The "Tell me about this project" answer)
"I built a comprehensive Generative AI Financial Advisor using Python and Streamlit. It’s designed to act as a personal finance assistant that goes beyond simple API calls. I implemented a local Retrieval-Augmented Generation (RAG) pipeline using **FAISS** and open-source models like **Flan-T5-Large** to answer financial questions accurately based on a specific knowledge base. I also integrated data-driven tools like a Risk Profiler, an Article Summarizer, and a real-time Stock Tracker to provide a fully featured, full-stack FinTech application."

---

## 2. Architecture & Tech Stack Breakdown
When asked "How does it work?", explain the pipeline logically:
*   **Frontend/UI:** Built with **Streamlit** to create an interactive, multi-page web application (Risk Profiler, Chatbot, Summarizer, Stock Details).
*   **Vector Database (Memory):** Uses **FAISS** (Facebook AI Similarity Search) to index chunked text documents (like financial FAQs).
*   **Embeddings:** Uses HuggingFace's `all-MiniLM-L6-v2` to convert text into dense vector representations.
*   **Generation (LLM):** Instead of relying entirely on paid APIs like OpenAI, the core QA chain utilizes `google/flan-t5-large` via a local HuggingFace Seq2Seq pipeline.
*   **The RAG Pipeline:** `vectorstore.py` splits text files into 500-character chunks with a 50-character overlap. `qa_chain.py` takes the user's prompt, embeds it, retrieves the top 3 (`k=3`) most similar document chunks from FAISS, and injects them into the Flan-T5 prompt context for a grounded, hallucination-free answer.

---

## 3. Anticipated Interview Questions & How to Answer Them

### **Q1: Why did you use FAISS instead of a managed vector database like Pinecone or Weaviate?**
**A:** "Because my dataset (financial FAQs) was relatively static and fit easily into memory, setting up a heavy cloud infrastructure like Pinecone was overkill and introduced unnecessary latency and cost. FAISS provided extremely fast, in-memory similarity search locally, which was perfect for demonstrating the core RAG functionality. If I were to scale this to millions of financial documents, I would migrate the FAISS index to a persistent vector database."

### **Q2: Why did you choose chunk sizes of 500 characters with a 50-character overlap in your text splitter?**
**A:** "Financial texts are dense. If the chunk size is too large, the retrieved context dilutes the specific answer and confuses the LLM (especially smaller models like Flan-T5). If the chunk size is too small, the context loses its meaning. 500 characters strikes a good balance for isolating a specific financial concept, and the 50-character overlap ensures that if a key sentence falls directly on a chunk boundary, the context isn't severed."

### **Q3: How do you prevent the AI from giving dangerous or illegal financial advice?**
**A:** "This is exactly why I implemented a RAG architecture instead of just prompting a raw LLM. By retrieving context from a strict, pre-approved vector database, the LLM is explicitly instructed via the prompt template to *only* answer based on the provided context (`Answer the question based on the context below`). If the user asks a wild investment question, the retriever won't find relevant context, mitigating hallucinations. In a production setting, I would also add a guardrail layer to block sensitive inputs."

### **Q4: Why did you use `google/flan-t5-large` instead of GPT-4?**
**A:** "While GPT-4 is powerful, relying strictly on commercial APIs hides a lot of the actual engineering. By using an open-weights model like Flan-T5 and local HuggingFace embeddings (`all-MiniLM`), I demonstrated the ability to orchestrate a fully localized ML pipeline. This also addresses data privacy—a massive concern in FinTech—because sensitive financial queries don't have to be sent over the internet to a third party."

### **Q5: What was the biggest challenge in building this?**
**A:** "Tuning the RAG prompt and the retrieval metric (`k=3`). Initially, the LLM would sometimes ignore the retrieved context and rely on its pre-trained knowledge. I had to strictly engineer the prompt template to explicitly force the model to look at the injected context block before generating its output. Additionally, ensuring that `LangChain` correctly routed the data between the FAISS index, the embedding model, and the generation pipeline required careful debugging."
