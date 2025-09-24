import csv, string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from evaluation import evaluate_and_plot_cm
from preprocessing import read_dataset, Emotions



class NaiveBayesClassifier:
    def __init__(self, idx2label, max_features=10000):
        self.idx2label = idx2label
        self.label2idx = {v: k for k, v in idx2label.items()}
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=4,
            max_df=0.95,
            max_features=max_features,
            stop_words='english',
            sublinear_tf=True
        )
        self.model = MultinomialNB()

    def prepare_texts(self, instances):
        return [inst.text for inst in instances]

    def fit(self, X, y):
        texts = self.prepare_texts(X)
        X_tfidf = self.vectorizer.fit_transform(texts)
        self.model.fit(X_tfidf, y)

    def predict(self, X):
        texts = self.prepare_texts(X)
        X_tfidf = self.vectorizer.transform(texts)
        return self.model.predict(X_tfidf)

    def predict_proba(self, X):
        texts = self.prepare_texts(X)
        X_tfidf = self.vectorizer.transform(texts)
        return self.model.predict_proba(X_tfidf)

    def evaluate(self, X_eval, y_eval, name=""):
        y_pred = self.predict(X_eval)
        accuracy = accuracy_score(y_eval, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_eval, y_pred, average=None, labels=range(len(self.idx2label))
        )
        macro_f1 = f1_score(y_eval, y_pred, average='macro')

        print("\nEvaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        for i in range(len(self.idx2label)):
            label_name = self.idx2label[i]
            print(f"{label_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")
     


def run_naive_bayes(train_path, test_path, labels):
    LABEL_LIST = sorted(labels)
    label2idx = {label: i for i, label in enumerate(LABEL_LIST)}
    idx2label = {i: label for label, i in label2idx.items()}

    print("Loading data...")
    train_data = read_dataset(train_path, labels)
    test_data = read_dataset(test_path, labels)
    print(f"Loaded {len(train_data)} training and {len(test_data)} test samples.")

    y_train = np.array([label2idx[inst.emotion] for inst in train_data])
    y_test  = np.array([label2idx[inst.emotion] for inst in test_data])

    print("Training Naive Bayes classifier...")
    model = NaiveBayesClassifier(idx2label=idx2label, max_features=2100)
    model.fit(train_data, y_train)

    print("Test performance:")
    y_pred = model.predict(test_data)
    evaluate_and_plot_cm(y_test, y_pred, LABEL_LIST, title="Naive Bayes Confusion Matrix", save_path="confusion_matrix_naivebayes.png")

    # Example of predicting probabilities for a single instance
    if test_data:
        print("\n--- Example Prediction with Probabilities ---")
        sample_instance = test_data[0]
        print(f"Sample Text: '{sample_instance.text}'")
        print(f"True Label: {sample_instance.emotion}")
        pred_probs = model.predict_proba([sample_instance])[0]
        predicted_label_idx = np.argmax(pred_probs)
        predicted_label = model.idx2label[predicted_label_idx]
        print(f"Predicted Label: {predicted_label}")
        print("Probabilities for each class:")
        for i, prob in enumerate(pred_probs):
            label_name = model.idx2label[i]
            print(f"  {label_name}: {prob:.4f}")

if __name__ == "__main__":
    LABELS = {"joy", "anger", "fear", "disgust", "sadness", "shame", "guilt"}
    train_path = "C:/Users/24350/env-nlp/emotional-damage/final_submission/train_modi.csv"
    test_path = "C:/Users/24350/env-nlp/emotional-damage/final_submission/isear-test.csv"
    run_naive_bayes(train_path, test_path, LABELS)
