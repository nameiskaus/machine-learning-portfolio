import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
import os
import streamlit as st

from src.stock_utils import get_stock_price, plot_stock_performance, get_company_financials



# Sidebar for stock selection
st.sidebar.header("Stock Information")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)")

if ticker_symbol:
    # Fetch current stock price
    price = get_stock_price(ticker_symbol)
    st.subheader(f"📊 Current Stock Price: {ticker_symbol}")
    st.write(f"${price:.2f}")

    # Display performance chart for the past month
    st.subheader(f"📈 {ticker_symbol} Performance (Last 1 Month)")
    fig = plot_stock_performance(ticker_symbol)
    st.plotly_chart(fig)

    # Fetch and display company financials
    st.subheader(f"🏢 {ticker_symbol} Financials")
    financials = get_company_financials(ticker_symbol)
    st.write(financials)