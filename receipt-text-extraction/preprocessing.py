import pandas as pd
import re

# Load the CSV
df = pd.read_csv("receipts_cleaned.csv")

# Function to filter meaningful product names
def is_meaningful(product):
    product_lower = product.lower()

    # Remove if contains certain keywords
    if any(keyword in product_lower for keyword in [
        "total", "subtotal", "tax", "debit", "credit", "change", "tend", "lb", "ib", "check", "member", "iotal"
    ]):
        return False

    # Remove if more than 50% of characters are digits
    digits = sum(char.isdigit() for char in product)
    if digits > len(product) * 0.5:
        return False

    # Remove if less than 2 alphabetic words
    word_count = len(re.findall(r'\b[a-zA-Z]{2,}\b', product))
    if word_count < 2:
        return False

    return True

# Apply the filter
df = df[df['Product'].apply(is_meaningful)]

# Save final cleaned data
df.to_csv("receipts_final_cleaned.csv", index=False)
print("✅ Final cleaned data saved to receipts_final_cleaned.csv")
