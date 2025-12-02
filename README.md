📄 **Automatic Extraction of Key Information from Research Papers using NLP & LayoutLM**

This project focuses on automatically extracting important information from research papers in PDF format using OCR, NLP & document layout modeling. The system processes research papers, recognizes text, understands layout patterns, and extracts structured information such as:

✔ Title

✔ Abstract

✔ Keywords

✔ Named Entities

✔ Token labels from LayoutLM

It aims to automate literature analysis and reduce manual reading effort for researchers, students, and analysts.

🎯 **Project Objective**
- Automate extraction of key data from research PDFs
- Reduce manual reading time
- Convert unstructured PDF text → structured JSON
- Assist research scholars in faster literature review



🔥 **Features**

| Feature                     | Description                                           |
| --------------------------- | ----------------------------------------------------- |
| 📄 PDF to Image Conversion  | Converts pages to image format using `pdf2image`      |
| 🔍 OCR Text Extraction      | Uses `pytesseract` to extract raw text from PDF pages |
| 🧠 Layout-aware Processing  | Uses LayoutLM model to understand document structure  |
| 🏷 Named Entity Recognition | Detects people, places, dates, organizations, etc.    |
| 📦 Structured Output        | Extracted data returned in JSON format                |
| ⚙️ Page-wise Processing     | Prevents RAM crashes by processing pages individually |



🧠 **Technology Stack**

| Category             | Tools Used          |
| -------------------- | ------------------- |
| OCR                  | Pytesseract         |
| NLP                  | SpaCy, Transformers |
| Layout Model         | LayoutLM            |
| Preprocessing        | pdf2image, PIL      |
| Backend / Processing | Python              |
| Optional UI          | Streamlit           |


🚀 **Installation (Google Colab Recommended)**
1. Install required libraries
   pip install pytesseract pdf2image transformers spacy Pillow
   python -m spacy download en_core_web_sm

2. Install Poppler (required for pdf2image)
   apt-get install poppler-utils

   
🔧 **How to Run in Google Colab**

    from google.colab import files
    uploaded = files.upload()
    pdf_path = list(uploaded.keys())[0]
    result = extract_info(pdf_path)
    import json
    print(json.dumps(result, indent=2))
