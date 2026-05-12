import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Page configuration for a centered layout
st.set_page_config(page_title="GenAI Financial Advisor", layout="wide")

# Custom CSS for more refined styling with black background and specific font colors
st.markdown(
    """
    <style>
        body {
            background-color: #1e1e1e; /* Black background */
            color: white; /* Default text color to white */
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #FFFFFF; /* Orange for title */
            text-align: center;
        }
        .subtitle {
            font-size: 20px;
            color: #FFFFFF; 
            margin-bottom: 30px;
        }
        .content-box {
            background-color: #2c2c2c; /* Dark grey background for content */
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
        }
        .sidebar-text {
            font-size: 18px;
            color: #FFA500; /* Orange for sidebar text */
            margin-top: 20px;
        }
        .list-items {
            margin-left: 20px;
            margin-bottom: 10px;
        }
        .list-items li {
            font-size: 16px;
            color: #FFFFFF; 
        }
        .header-section {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            background-color: #FFFFFF; 
            color: white;
            border-radius: 10px 10px 0 0;
        }
        .header-section h2 {
            font-size: 28px;
            margin: 0;
        }
        .description {
            font-size: 18px;
            line-height: 1.6;
            margin: 20px 0;
            text-align: justify;
            color: #FFA500; /* Orange for description */
        }
        .card {
            background-color: #333333; /* Dark grey background for cards */
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
        }
        .card h3 {
            margin-top: 0;
            color: #FFFFFF; /* Dark Blue for card titles */
        }
        .card p {
            color: #FFA500; /* Orange for card text */
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Page title with professional style
st.markdown('<div class="title">🏠 Welcome to the GenAI Financial Advisor</div>', unsafe_allow_html=True)

# Section for page description and features
st.markdown(
    """
    <div class="description">
        Welcome to your personal financial assistant powered by GenAI. This platform helps you manage your financial 
        well-being with advanced AI tools. Explore the following features:
    </div>
    """,
    unsafe_allow_html=True,
)

# Feature Cards with rounded corners and shadows
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="card"><h3>Generate Risk Profile</h3><p>Understand your financial risk tolerance and get customized investment strategies.</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card"><h3>💬 Chat with Financial FAQ Bot</h3><p>Ask financial questions and get immediate responses based on a knowledge base of FAQs.</p></div>', unsafe_allow_html=True)

# Additional feature cards
col3, col4 = st.columns(2)
with col3:
    st.markdown('<div class="card"><h3>Article Summarizer</h3><p>Get quick summaries of financial articles from trusted sources.</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="card"><h3>Stock Details</h3><p>Track the latest stock market data and trends to make informed investment decisions.</p></div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card"><h3>Personal Finance Tip Generator</h3><p>Receive personalized finance tips to improve savings, budgeting, and investment strategies.</p></div>', unsafe_allow_html=True)

