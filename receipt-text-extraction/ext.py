import os
import re
import csv
import easyocr
from PIL import Image

# Folder containing receipt images
receipt_folder = "/Users/aadyamohanty/Desktop/byteme/images"  # Update this to your path
output_csv = "receipts.csv"

# Set up OCR reader
reader = easyocr.Reader(['en'])

def clean_product_name(name):
    # Remove long numbers (product codes)
    name = re.sub(r'\b\d{6,}\b', '', name)
    # Remove unwanted punctuation
    name = re.sub(r'[\'\"\,]+', ' ', name)
    # Remove stray characters
    name = re.sub(r'[^a-zA-Z0-9\s\-\/]', '', name)
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name)
    return name.strip().title()

def is_valid_price(text):
    return re.fullmatch(r"\d+\.\d{2}", text.strip()) is not None

def is_line_relevant(text):
    keywords = ['total', 'subtotal', 'tax', 'debit', 'credit', 'tend', 'change']
    return not any(k in text.lower() for k in keywords)

# Prepare output CSV
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Product', 'Price'])

    for img_file in os.listdir(receipt_folder):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        img_path = os.path.join(receipt_folder, img_file)
        print(f"Processing {img_file}...")
        results = reader.readtext(img_path)

        # Store results as list of dicts with bounding box + text
        lines = [
            {'bbox': bbox, 'text': text.strip(), 'conf': conf}
            for (bbox, text, conf) in results
            if conf > 0.3 and text.strip() != ""
        ]

        # Sort top-to-bottom by y-coordinate
        lines.sort(key=lambda x: x['bbox'][0][1])

        for line in lines:
            words = line['text'].split()
            if not is_line_relevant(line['text']):
                continue

            for i, word in enumerate(words):
                if is_valid_price(word):
                    product = ' '.join(words[:i])
                    price = word
                    cleaned = clean_product_name(product)
                    if cleaned and len(cleaned.split()) > 0:
                        writer.writerow([cleaned, price])
                    break  # Move to next line after first price

print(f"\n✅ Done! Clean receipt data saved to: {output_csv}")
