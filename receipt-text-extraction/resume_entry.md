**Automated Receipt OCR Pipeline** | Python, EasyOCR, Pandas, Pillow, Regex  
Developed a computer vision pipeline to extract and digitize structured product and pricing data from raw receipt images.  
- Implemented an EasyOCR text recognition system, parsing bounding box coordinates to geometrically align receipt lines from top to bottom.  
- Engineered custom Regex-based string cleaning and heuristic filtering to automatically eliminate receipt metadata (subtotals, taxes, card details).  
- Built a Pandas data processing module to validate extracted prices and format unstructured OCR text into clean, usable CSV datasets.  
- Optimized data quality via multi-layered validation logic, systematically filtering out entries with extreme alphanumeric noise or insufficient word counts.
