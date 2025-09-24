import csv
import string
import os
from collections import Counter
from evaluation import calculate_metrics
# Only keep these seven emotion labels
LABELS = {"joy", "anger", "fear", "disgust", "sadness", "shame", "guilt"}

def read_labels(path, allowed_labels):
    labels = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row.get("label", "").strip().lower()
            if lbl in allowed_labels:
                labels.append(lbl)
    return labels

# 1. Load training and validation labels
y_train = read_labels("C:/Users/24350/env-nlp/emotional-damage/final_submission/train_modi.csv", LABELS)
y_val   = read_labels("C:/Users/24350/env-nlp/emotional-damage/final_submission/isear-val.csv", LABELS)

# 2. Find the most frequent label in the training set
major_label, _ = Counter(y_train).most_common(1)[0]

# 3. Predict the same label for all validation instances
y_pred = [major_label] * len(y_val)

# 4. Compute metrics using the evaluation functions
metrics = calculate_metrics(y_val, y_pred, len(LABELS))
accuracy = metrics['accuracy']
macro_f1 = metrics['macro_f1']
pred_counts = Counter(y_pred)

# 5. Print overall metrics
print("Dummy Classifier Performance:")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  Macro F1 : {macro_f1:.4f}\n")

# 6. Since we always predict the same label:
print(f"Always-predicted label: {major_label!r} ({pred_counts[major_label]} samples, "
      f"{pred_counts[major_label]/len(y_pred):.1%} of total)")