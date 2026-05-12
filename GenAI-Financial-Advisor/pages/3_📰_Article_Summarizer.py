import streamlit as st
from bs4 import BeautifulSoup
import requests
from transformers import pipeline

st.set_page_config(page_title="Article Summarizer", layout="centered")

# Initialize Hugging Face summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to extract paragraphs and clean up
def extract_paragraphs_from_html(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        for unwanted_tag in soup(["script", "style", "meta", "header", "footer", "nav"]):
            unwanted_tag.decompose()

        paragraphs = soup.find_all("p")
        article_content = "\n".join([p.get_text() for p in paragraphs if p.get_text().strip()])

        if len(article_content) < 200:
            return "❌ Article content is too short."

        return article_content
    except Exception as e:
        return f"❌ Error during content extraction: {str(e)}"

# Summarize the article
def summarize_article(text):
    try:
        input_text = text[:2000]
        summary = summarizer(input_text, max_length=100, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"❌ Error during summarization: {str(e)}"

# Streamlit app
st.title("📰 Finance News Summarizer")

url = st.text_input("Enter Article URL")

if url:
    content = extract_paragraphs_from_html(url)
    if "❌" not in content:
        st.subheader("📊 Summary:")
        st.write(summarize_article(content))
    else:
        st.warning(content)
