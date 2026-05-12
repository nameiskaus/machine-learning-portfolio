import feedparser
import random
from transformers import pipeline
import requests
import trafilatura
from bs4 import BeautifulSoup
import streamlit as st
from newspaper import Article  # Add this for the newspaper3k method

# Initialize Hugging Face summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Google News RSS feed URL for finance
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q=finance&hl=en-US&gl=US&ceid=US:en"

def fetch_top_headlines_rss(limit=5):
    feed = feedparser.parse(GOOGLE_NEWS_RSS)
    if not feed.entries:
        raise Exception("No news articles found in RSS feed.")
    return [{"title": entry.title, "link": entry.link} for entry in feed.entries[:limit]]

def fetch_random_article():
    feed = feedparser.parse(GOOGLE_NEWS_RSS)
    if not feed.entries:
        raise Exception("No news articles found in RSS feed.")
    random_entry = random.choice(feed.entries)
    return {"title": random_entry.title, "link": random_entry.link}

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        if len(article.text) > 200:
            return article.text
        else:
            return "❌ Article content is too short."
    except Exception as e:
        return f"❌ Error during extraction: {str(e)}"

def fetch_article_content_using_trafilatura(url):
    try:
        # Follow redirect from Google News
        response = requests.get(url, timeout=10, allow_redirects=True)
        final_url = response.url
        print(f"🔗 Final resolved URL: {final_url}")

        # Fetch the page HTML
        downloaded = trafilatura.fetch_url(final_url)

        if downloaded:
            extracted = trafilatura.extract(downloaded)
            if extracted:
                print(f"📄 Extracted content length: {len(extracted)}")
                return extracted
            else:
                print("⚠️ Extraction returned None.")
                return "❌ Article content could not be extracted or is too short."
        else:
            print("⚠️ Failed to fetch HTML with trafilatura.")
            return "❌ Failed to fetch article HTML."

    except Exception as e:
        print(f"❌ Exception in fetch_article_content: {e}")
        return f"❌ Error fetching article: {str(e)}"


def summarize_article(text):
    try:
        # Limit input to ~1024 tokens (about 1300–1500 characters)
        max_input_length = 2000
        input_text = text[:max_input_length]

        summary = summarizer(
            input_text,
            max_new_tokens=150,  # ✅ Replace max_length
            min_length=50,
            do_sample=False
        )

        return summary[0]['summary_text']
    except Exception as e:
        return f"❌ Error during summarization: {str(e)}"


# Streamlit App
def app():
    st.title("Finance News Summarizer")

    # Fetch a random article
    article = fetch_random_article()
    st.subheader(f"📰 Title: {article['title']}")
    st.markdown(f"🔗 [Read Full Article]({article['link']})")

    # Fetch the full article content before summarizing
    article_content = fetch_article_content_using_trafilatura(article['link'])
    
    # If content is extracted successfully, summarize
    if "❌" not in article_content:
        
        st.subheader("📊 Summary:")
        summary = summarize_article(article_content)
        st.write(summary)  # Display the summary in Streamlit
    else:
        st.warning("⚠️ Unable to extract or summarize the article.")

if __name__ == "__main__":
    app()
