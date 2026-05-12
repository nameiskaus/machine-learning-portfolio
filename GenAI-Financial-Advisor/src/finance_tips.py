import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load environment variables from .env file
load_dotenv()

# Load the model using Hugging Face's pipeline (once)
def load_model(model_name="google/flan-t5-large"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)

# Load model once globally to avoid reloading
flan_pipe = load_model()

# Generate budgeting advice: rule-based + AI-generated
def generate_budgeting_advice(income, needs, wants, savings_goal):
    fixed_expenses = needs + wants
    remaining = income - fixed_expenses
    rule_based_tips = []

    # Basic budgeting rule logic
    if needs > income * 0.5:
        rule_based_tips.append("Your essential expenses exceed 50% of your income. Consider downsizing housing, switching utility providers, or adjusting grocery budgets.")
    if wants > income * 0.3:
        rule_based_tips.append("Your discretionary spending is high. Reduce non-essentials like shopping, entertainment, or subscriptions.")
    if remaining < savings_goal:
        rule_based_tips.append("You're not meeting your savings goal. Automate savings or reduce small recurring expenses.")
    if not rule_based_tips:
        rule_based_tips.append("You're on track! Consider investing excess funds in low-risk options or emergency savings.")

    # Instructional prompt for FLAN-T5
    prompt = f"""
    Generate 3 beginner-friendly personal finance tips for someone with the following details:

    - Monthly Income: {income}
    - Monthly Spending on Needs: {needs}
    - Monthly Spending on Wants: {wants}
    - Monthly Savings Goal: {savings_goal}

    Be specific and practical in your suggestions.
    """

    try:
        ai_response = flan_pipe(prompt)[0]['generated_text']
    except Exception as e:
        ai_response = f"FLAN-T5 could not generate tips due to: {e}"

    return rule_based_tips, ai_response
