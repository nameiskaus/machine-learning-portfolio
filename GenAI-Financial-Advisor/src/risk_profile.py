import plotly.express as px
import streamlit as st

questions = {
    "Number of Dependents": {"0": 3, "1–2": 2, "3+": 1},
    "Expected Annual Return": {"5–7%": 1, "8–12%": 2, "13%+": 3},
    "Risk Tolerance": {
        "I panic when markets fall": 1,
        "I wait and observe": 2,
        "I invest more during dips": 3
    },
    "Reaction to Portfolio Drop": {
        "I sell immediately": 1,
        "I hold and wait": 2,
        "I buy more at lower prices": 3
    },
    "Liquidity Horizon": {
        "<1 year": 1,
        "1–5 years": 2,
        "More than 5 years": 3
    },
    "Preferred Portfolio Type": {
        "Fixed Income only": 1,
        "Balanced (Debt & Equity)": 2,
        "Equity-heavy with some debt": 3
    },
    "Investment Goal": {
        "Capital preservation": 1,
        "Steady long-term growth": 2,
        "Aggressive wealth creation": 3
    },
    "Your Age": {
        "Above 60": 1,
        "30 to 60": 2,
        "Below 30": 3
    }
}

def calculate_profile():
    total_score = 0
    responses = {}
    st.markdown("### 📝 Complete the questionnaire below:")

    for i, (q, options) in enumerate(questions.items()):
        selected = st.radio(
            f"**{q}**",
            options=list(options.keys()),
            index=None,
            key=f"q_{i}"
        )
        responses[q] = selected

    # Check if any answer is missing
    if None in responses.values():
        return None, responses
    else:
        for q, answer in responses.items():
            total_score += questions[q][answer]
        return total_score, responses

def get_profile(score):
    if score <= 5:
        return "Secure", {"Debt": 90, "Equity": 10}
    elif score <= 10:
        return "Conservative", {"Debt": 75, "Equity": 25}
    elif score <= 15:
        return "Moderate", {"Debt": 50, "Equity": 50}
    elif score <= 20:
        return "Growth", {"Debt": 30, "Equity": 70}
    else:
        return "Aggressive", {"Debt": 10, "Equity": 90}

def render_pie_chart(allocation):
    fig = px.pie(
        names=list(allocation.keys()),
        values=list(allocation.values()),
        title="📈 Suggested Asset Allocation",
        color_discrete_sequence=px.colors.sequential.Agsunset
    )
    st.plotly_chart(fig)
