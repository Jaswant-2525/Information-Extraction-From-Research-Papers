import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Load JSON outputs
with open("output.json", "r", encoding="utf-8") as f:
    pred = json.load(f)
with open("ground_truth.json", "r", encoding="utf-8") as f:
    gt = json.load(f)

# 1. Core Term Frequency – Bar Chart
def plot_core_term_frequency(gt_terms, pred_terms):
    combined = gt_terms + pred_terms
    freq = Counter(term.lower() for term in combined)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(freq.keys(), freq.values(), color="#1f77b4")  # Blue
    plt.title("Core Term Frequency", fontsize=14)
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# 2. Section Extraction – Pie Chart
def plot_section_extraction(gt_sections, pred_sections):
    correct, partial, missed = 0, 0, 0
    for sec in gt_sections:
        if sec in pred_sections:
            if len(pred_sections[sec]) > 50:
                correct += 1
            else:
                partial += 1
        else:
            missed += 1

    labels = ['Correct', 'Partial', 'Missed']
    values = [correct, partial, missed]
    colors = ['#2ca02c', '#ff7f0e', '#d62728']  # Green, Orange, Red

    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=140, textprops={'fontsize': 12})
    plt.title("Section Extraction Quality", fontsize=14)
    plt.tight_layout()
    plt.show()

# 3. NER Histogram
def plot_ner_histogram(pred_entities):
    label_counts = Counter(label for _, label in pred_entities)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(label_counts.keys(), label_counts.values(), color='#9467bd')  # Purple
    plt.title("Named Entity Label Distribution", fontsize=14)
    plt.ylabel("Count")
    plt.xlabel("Entity Label")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval),
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

# 4. Confusion Matrix – NER
def plot_ner_confusion_matrix(gt_entities, pred_entities):
    gt_dict = {ent["text"].lower(): ent["label"] for ent in gt_entities}
    pred_dict = {ent[0].lower(): ent[1] for ent in pred_entities}

    common_texts = set(gt_dict.keys()) & set(pred_dict.keys())
    if not common_texts:
        print("No common named entities found for confusion matrix.")
        return

    y_true = [gt_dict[text] for text in common_texts]
    y_pred = [pred_dict[text] for text in common_texts]
    labels = sorted(set(y_true + y_pred))

    encoder = LabelEncoder().fit(labels)
    y_true_enc = encoder.transform(y_true)
    y_pred_enc = encoder.transform(y_pred)

    cm = confusion_matrix(y_true_enc, y_pred_enc)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix – Named Entity Recognition", fontsize=14)
    plt.tight_layout()
    plt.show()

# 5. Simulated OCR Time vs Text Length – Line Plot
def plot_ocr_time_vs_length():
    pages = [1, 2, 3, 4, 5]
    lengths = [1000, 1200, 950, 1100, 1300]
    times = [1.8, 2.0, 1.7, 2.1, 2.5]

    plt.figure(figsize=(10, 6))
    plt.plot(pages, lengths, label="Text Length", marker='o', color="#17becf")
    plt.plot(pages, times, label="OCR Time (s)", marker='s', color="#ff9896")
    plt.xlabel("Page Number")
    plt.ylabel("Value")
    plt.title("OCR Processing Time vs Text Length", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run all
plot_core_term_frequency(gt.get("core_terms", []), pred.get("core_terms", []))
plot_section_extraction(gt.get("sections", {}), pred.get("sections", {}))
plot_ner_histogram(pred.get("named_entities", []))
plot_ner_confusion_matrix(gt.get("named_entities", []), pred.get("named_entities", []))
plot_ocr_time_vs_length()
