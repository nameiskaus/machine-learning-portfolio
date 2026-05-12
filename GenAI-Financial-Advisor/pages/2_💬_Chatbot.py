import streamlit as st
from qa_chain import load_qa_chain
from transformers import pipeline

# Initialize the GPT model for explanation (for layman terms)
explanation_model = pipeline("text2text-generation", model="google/flan-t5-large")

# Set page configuration
st.set_page_config(page_title="GenAI Financial Chatbot", layout="centered")

# Set up the page title and description
st.title("💬 Financial FAQ Chatbot")
st.markdown("Ask a question based on your uploaded financial FAQs or ask about a finance term in simple terms.")

# Load QA chain for FAQ-based questions
try:
    qa_chain = load_qa_chain()
except Exception as e:
    st.error(f"❌ Failed to load QA chain: {e}")
    st.stop()

# Initialize session state
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("input_text", "")
st.session_state.setdefault("mode", "faq")  # Track the mode (FAQ or explanation)

# Option for user to choose the mode
mode = st.radio("Select query type:", ("FAQ", "Explain a term"))

# Input row: text box and submit button
col1, col2 = st.columns([6, 1])
with col1:
    query = st.text_input(
        label="💡 Ask your question:",
        value=st.session_state.input_text,
        label_visibility="collapsed",
        key="input_text_box"
    )
with col2:
    submitted = st.button("Submit")

# Process query based on mode (FAQ or explanation)
if submitted and query.strip():
    with st.spinner("Generating answer..."):
        if mode == "FAQ":
            # Use the FAQ-based QA chain
            answer = qa_chain(query.strip())
        elif mode == "Explain a term":
            # Use the GPT model for explanation in layman terms
            prompt = f"Explain {query.strip()} in simple terms to a beginner finance student."
            answer = explanation_model(prompt)[0]['generated_text']

    # Append the response to chat history
    st.session_state.chat_history.append({
        "question": query.strip(),
        "answer": answer
    })

    # Reset input text and re-run the app to show updated chat history
    st.session_state.input_text = ""
    st.rerun()

# Display chat history with different styling
for chat in reversed(st.session_state.chat_history):
    if mode == "FAQ":
        st.markdown(f"""
            <div style='background-color:#2c2f33;padding:10px;border-radius:10px;margin-bottom:10px;'>
                <p style='color:#00acee;'><strong>You:</strong> {chat['question']}</p>
                <p style='color:#b9fbc0;'><strong>Bot:</strong> {chat['answer']}</p>
            </div>
        """, unsafe_allow_html=True)
    elif mode == "Explain a term":
        st.markdown(f"""
            <div style='background-color:#2c2f33;padding:10px;border-radius:10px;margin-bottom:10px;'>
                <p style='color:#00acee;'><strong>You asked about:</strong> {chat['question']}</p>
                <p style='color:#b9fbc0;'><strong>Bot's explanation:</strong> {chat['answer']}</p>
            </div>
        """, unsafe_allow_html=True)