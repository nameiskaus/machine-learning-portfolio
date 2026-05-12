# pages/1_📊_Risk_Profiler.py

import streamlit as st
from src.risk_profile import calculate_profile, get_profile, render_pie_chart

st.set_page_config(page_title="GenAI Financial Advisor", layout="centered")
st.title("📊 Advanced Risk Profiler")

score, responses = calculate_profile()

if st.button("Get My Profile"):
    if score is None:
        st.warning("⚠️ Please answer all questions before generating your profile.")
    else:
        profile, allocation = get_profile(score)
        st.success(f"**Your Risk Profile: {profile}**")
        st.info(f"**Score:** {score} / 24")
        render_pie_chart(allocation)
