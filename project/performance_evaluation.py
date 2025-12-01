import json
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_named_entities(gt_entities, pred_entities):
    print("\nNamed Entity Recognition (NER) Evaluation:")

    # Convert list of tuples to set of strings for matching
    gt_set = set((ent["text"].strip().lower(), ent["label"]) for ent in gt_entities)
    pred_set = set((ent[0].strip().lower(), ent[1]) for ent in pred_entities)

    true_positives = len(gt_set & pred_set)
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

def evaluate_token_labels(gt_tokens, pred_tokens):
    gt_labels = [label for _, label in gt_tokens]
    pred_labels = [label for _, label in pred_tokens]
    print("\nToken Classification Evaluation:")
    print(classification_report(gt_labels, pred_labels))

def evaluate_sections(gt_sections, pred_sections):
    gt_keys = set(k.lower() for k, v in gt_sections.items() if v.strip())
    pred_keys = set(k.lower() for k, v in pred_sections.items() if v.strip())
    tp = len(gt_keys & pred_keys)
    fp = len(pred_keys - gt_keys)
    fn = len(gt_keys - pred_keys)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    print("\nSection Extraction Evaluation:")
    print(f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1-Score: {f1:.2f}")

def evaluate_core_terms(gt_terms, pred_terms):
    gt_set = set(gt_terms)
    pred_set = set(pred_terms)
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    print("\nCore Terms Evaluation:")
    print(f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1-Score: {f1:.2f}")

def main(gt_file='ground_truth.json', pred_file='output.json'):
    gt = load_json(gt_file)
    pred = load_json(pred_file)

    if "named_entities" in gt and "named_entities" in pred:
        evaluate_named_entities(gt["named_entities"], pred["named_entities"])

    if "tokens_with_labels" in gt and "tokens_with_labels" in pred:
        evaluate_token_labels(gt["tokens_with_labels"], pred["tokens_with_labels"])

    if "sections" in gt and "sections" in pred:
        evaluate_sections(gt["sections"], pred["sections"])

    if "core_terms" in gt and "core_terms" in pred:
        evaluate_core_terms(gt["core_terms"], pred["core_terms"])

if __name__ == "__main__":
    main()

