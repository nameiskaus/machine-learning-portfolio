import streamlit as st
from src.local_ai import load_model, generate_local_tips
from src.finance_tips import generate_budgeting_advice

# Load model once
@st.cache_resource
def get_model():
    return load_model("google/flan-t5-large")

pipe = get_model()

# User input form
with st.form("finance_form"):
    income = st.number_input("Monthly Income", min_value=0.0, step=100.0)
    needs = st.number_input("Monthly Needs (Rent, Bills)", min_value=0.0, step=50.0)
    wants = st.number_input("Monthly Wants (Shopping, Dining)", min_value=0.0, step=50.0)
    savings_goal = st.number_input("Monthly Savings Goal", min_value=0.0, step=50.0)
    submitted = st.form_submit_button("Generate Tips")

if submitted:
    # Rule-based budgeting tips
    rule_tips, _ = generate_budgeting_advice(income, needs, wants, savings_goal)
    
    st.subheader("📌 Rule-Based Advice")
    for tip in rule_tips:
        st.markdown(f"- {tip}")
    
    # AI-generated tips using FLAN-T5 XL
    st.subheader("🤖 AI-Generated Tips ")
    with st.spinner("Thinking..."):
        response = generate_local_tips(pipe, income, needs, wants, savings_goal)
        st.markdown(response)
