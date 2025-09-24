import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
from evaluation import calculate_metrics, print_metrics, evaluate_and_plot_cm
from preprocessing import read_dataset

train_p = r"C:/Users/24350/env-nlp/emotional-damage/final_submission/train_modi.csv"
val_p = r"C:/Users/24350/env-nlp/emotional-damage/final_submission/isear-val.csv"
test_p = r"C:/Users/24350/env-nlp/emotional-damage/final_submission/isear-test.csv"
paths = {"train": train_p, "val": val_p, "test": test_p}
LABELS = {"joy", "anger", "fear", "disgust", "sadness", "shame", "guilt"}
label_list = sorted(LABELS)
label2idx = {l: i for i, l in enumerate(sorted(LABELS))}
idx2label = {i: l for l, i in label2idx.items()}

def load(split):
    samples = read_dataset(paths[split], LABELS)
    texts = [s.text for s in samples]
    labels = [label2idx[s.emotion] for s in samples]
    return texts, labels

tr_texts, tr_labels = load("train")
va_texts, va_labels = load("val")
te_texts, te_labels = load("test")

# TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
X_tr = vectorizer.fit_transform(tr_texts).toarray()
X_va = vectorizer.transform(va_texts).toarray()
X_te = vectorizer.transform(te_texts).toarray()

# to TensorDataset
def make_loader(X, y, bs, shuffle=False):
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    return DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=shuffle)

train_loader = make_loader(X_tr, tr_labels, bs=64, shuffle=True)
val_loader   = make_loader(X_va, va_labels, bs=64)
test_loader  = make_loader(X_te, te_labels, bs=64)

# FFNN
class FFNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FFNN(input_dim=X_tr.shape[1], num_classes=len(LABELS), hidden_dim=128).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

if __name__ == "__main__":
    print("=== Training: 8 epochs with shuffle ===")
    for epoch in range(1, 9):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Train Loss: {total_loss/len(train_loader):.4f}")

        # val
        model.eval()
        preds, golds = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb)
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                golds.extend(yb.numpy())
        metrics = calculate_metrics(np.array(golds), np.array(preds), len(LABELS))
        print_metrics(metrics, f"Epoch {epoch}", idx2label)

    # test
    print("\n=== Final Evaluation on Test Set ===")
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            golds.extend(yb.numpy())
    metrics = calculate_metrics(np.array(golds), np.array(preds), len(LABELS))
    print_metrics(metrics, "Test", idx2label)

    y_pred = []
    model.eval()
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            out = model(xb)
            y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())

    # visualized evaluation
    evaluate_and_plot_cm(te_labels, y_pred, label_list, title="FFNN Confusion Matrix", save_path="confusion_matrix_perceptron.png")
