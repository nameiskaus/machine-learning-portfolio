
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
import os
# Function to get stock price using yfinance
def get_stock_price(ticker: str):
    stock = yf.Ticker(ticker)
    price = stock.history(period="1d")['Close'][0]
    return price

# Function to plot stock performance for the last month
def plot_stock_performance(ticker: str):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo")
    
    fig = go.Figure(data=[go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name=f'{ticker} Performance'
    )])
    
    fig.update_layout(title=f"{ticker} - Past 1 Month Performance", xaxis_title="Date", yaxis_title="Price (USD)")
    return fig

# Function to get company financials (balance sheet)
def get_company_financials(ticker: str):
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    return balance_sheet
