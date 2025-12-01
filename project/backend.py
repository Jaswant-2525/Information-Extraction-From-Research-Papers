import pytesseract
from pdf2image import convert_from_path
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
from PIL import Image
import torch
import spacy
import tkinter as tk
from tkinter import filedialog
import json
import os
import csv
import nltk
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Downloads
nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

# Optionally configure Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def convert_to_layoutlm_boxes(x, y, w, h, width, height):
    x0 = int(1000 * x / width)
    y0 = int(1000 * y / height)
    x1 = int(1000 * (x + w) / width)
    y1 = int(1000 * (y + h) / height)
    return [x0, y0, x1, y1]

def load_model():
    model_name = "microsoft/layoutlm-base-uncased"
    tokenizer = LayoutLMTokenizer.from_pretrained(model_name)
    model = LayoutLMForTokenClassification.from_pretrained(model_name)
    return tokenizer, model

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        return ""

def extract_core_sections(text):
    sections = {"abstract": "", "introduction": "", "methodology": "", "results": "", "conclusion": ""}
    lower_text = text.lower()
    for section in sections.keys():
        index = lower_text.find(section)
        if index != -1:
            next_section_index = min([
                lower_text.find(s, index + len(section)) for s in sections if lower_text.find(s, index + len(section)) != -1
            ] + [len(text)])
            sections[section] = text[index:next_section_index].strip()
    return sections

def extract_core_terms(text, top_n=10):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]

    if not tokens:
        return ["No core terms found."]

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([" ".join(tokens)])
    except ValueError:
        return ["TF-IDF failed: empty input."]

    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [term for term, score in sorted_scores[:top_n]]

def extract_info(pdf_path, tokenizer, model):
    text_from_fitz = extract_text_from_pdf(pdf_path)

    if text_from_fitz:
        full_text = text_from_fitz
        sections = extract_core_sections(full_text)
        core_terms = extract_core_terms(full_text)
        doc = nlp(full_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return {
            "tokens_with_labels": [],
            "named_entities": entities,
            "sections": sections,
            "core_terms": core_terms
        }

    try:
        images = convert_from_path(pdf_path)
    except Exception as e:
        return {"error": f"PDF conversion failed: {e}"}

    all_extracted_tokens_with_labels = []
    all_raw_text = []

    for page_number, image in enumerate(images, start=1):
        width, height = image.size
        try:
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except Exception:
            continue

        words, boxes = [], []
        for i in range(len(ocr_data["text"])):
            text = str(ocr_data["text"][i]).strip()
            if text and ocr_data["conf"][i] > 0:
                x, y, w, h = (ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i])
                if w > 0 and h > 0:
                    words.append(text)
                    boxes.append(convert_to_layoutlm_boxes(x, y, w, h, width, height))

        if not words:
            continue

        try:
            encoded_inputs = tokenizer(words, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length")
            with torch.no_grad():
                outputs = model(**encoded_inputs)
            tokens = tokenizer.convert_ids_to_tokens(encoded_inputs['input_ids'][0])
            preds = torch.argmax(outputs.logits, dim=2)[0].tolist()
            page_tokens_with_labels = [
                (token, model.config.id2label[label_id])
                for token, label_id in zip(tokens, preds)
                if token not in tokenizer.all_special_tokens
            ]
            all_extracted_tokens_with_labels.extend(page_tokens_with_labels)
            all_raw_text.append(" ".join(words))
        except Exception:
            continue

    full_text = " ".join(all_raw_text).strip()

    if not full_text:
        return {
            "tokens_with_labels": [],
            "named_entities": [],
            "sections": {},
            "core_terms": ["No extractable text found in PDF."],
            "error": "OCR failed or PDF contained no readable text."
        }

    doc = nlp(full_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    sections = extract_core_sections(full_text)
    core_terms = extract_core_terms(full_text)

    return {
        "tokens_with_labels": all_extracted_tokens_with_labels,
        "named_entities": entities,
        "sections": sections,
        "core_terms": core_terms
    }

def main():
    root = tk.Tk()
    root.withdraw()
    pdf_path = filedialog.askopenfilename(title="Select a PDF file", filetypes=[("PDF files", "*.pdf")])
    if not pdf_path or not os.path.exists(pdf_path):
        print("Invalid or no file selected.")
        return

    result = extract_info(pdf_path, tokenizer, model)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Save as JSON
    json_filename = "output.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nExtracted data saved to {json_filename}")

    # Save named entities as CSV
    csv_filename = "named_entities.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Entity", "Label"])
        writer.writerows(result.get("named_entities", []))
    print(f"Named entities saved to {csv_filename}")

if __name__ == "__main__":
    print("Loading LayoutLM model...")
    try:
        tokenizer, model = load_model()
    except Exception as e:
        print(f"Model loading failed: {e}")
        exit()
    main()
