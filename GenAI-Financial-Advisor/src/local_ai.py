from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def load_model(model_name="google/flan-t5-large"):
    # Load the FLAN-T5 XL model (public and no token required)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # Create a text-generation pipeline with max_new_tokens set
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)
    return pipe

def generate_local_tips(pipe, income, needs, wants, savings_goal):
    prompt = f"""
You are a personal finance advisor.

A user has the following financial details:
- Monthly Income: {income}
- Spending on Needs: {needs}
- Spending on Wants: {wants}
- Monthly Savings Goal: {savings_goal}

Based on this information, generate 3 specific and practical financial tips to help them reduce spending, save more, and improve their financial health.

Write the tips in a numbered list. Do not repeat the input. Be detailed and helpful.
tips:
1. **Cut down on non-essentials**  
Your spending on wants is 30% of your income. Try reducing this by 10–15%, focusing on things like dining out or shopping. Reallocate the saved money into your savings goal.

2. **Automate savings**  
Set up an automatic monthly transfer of $800 directly into your savings account as soon as your paycheck hits. This makes saving effortless and ensures you meet your target.

3. **Reevaluate fixed costs**  
Check if you can cut down on fixed expenses like rent or utilities. Even small changes like switching insurance providers or renegotiating rent could free up additional funds for savings.

"""

    
    result = pipe(prompt)[0]['generated_text']
    return result
