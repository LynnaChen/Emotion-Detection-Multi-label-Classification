import csv
import numpy as np
from evaluation import calculate_metrics, print_metrics, evaluate_and_plot_cm
from preprocessing import read_dataset
import os

def build_vocabulary(samples):
    vocab = set()
    for sample in samples:
        vocab.update(sample.tokens)
    return sorted(vocab)

def vectorize(tokens, vocabulary): #vectorize the tokens
    token_set = set(tokens)
    return [1 if word in token_set else 0 for word in vocabulary]

def vectorize_dataset(samples, vocabulary=None):
    # If no vocabulary provided, build from samples
    if vocabulary is None:
        vocabulary = build_vocabulary(samples)
        print(f"Built new vocabulary, size: {len(vocabulary)}")
    else:
        print(f"Using provided vocabulary, size: {len(vocabulary)}")
    
    # Create vocabulary dictionary mapping words to indices
    vocab_dict = {word: idx for idx, word in enumerate(vocabulary)}
    
    # Create feature matrix using numpy
    n, d = len(samples), len(vocabulary)
    X = np.zeros((n, d), dtype=int)
    y = []
    
    for i, sample in enumerate(samples):
        y.append(sample.emotion)
        for tok in sample.tokens:
            j = vocab_dict.get(tok)
            if j is not None:
                X[i, j] = 1
    
    # Print debug information
    print(f"Number of samples: {len(samples)}")
    print(f"Feature vector dimension: {X.shape[1]}")
    
    return X, y, vocabulary

class MultiPerceptron:
    def __init__(self, input_dim, num_classes, idx2label):
        # Initialize weight matrix and class mappings
        self.W = np.zeros((num_classes, input_dim))  # Initialize weight matrix with zeros
        self.num_classes = num_classes  # Number of output classes
        self.idx2label = idx2label  # Class label mappings

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=10, shuffle_data=True):
        # Train model using perceptron
        n_samples = X_train.shape[0]  # Number of training samples
        
        best_epoch = 0  # Variable to store the best epoch
        best_val_accuracy = 0  # Variable to store the best validation accuracy

        for epoch in range(epochs):
            if shuffle_data:
                indices = np.random.permutation(n_samples)  # Shuffle indices for the data
                X_train_shuffled = X_train[indices]  # Shuffle the training data
                y_train_shuffled = y_train[indices]  # Shuffle the training labels
            else:
                X_train_shuffled, y_train_shuffled = X_train, y_train

            # Training the model (Perceptron algorithm)
            for i in range(n_samples):
                xi = X_train_shuffled[i]  # Sample data
                yi = y_train_shuffled[i]  # True label
                scores = self.W @ xi  # Compute scores for each class
                y_pred = np.argmax(scores)  # Predicted class (index of the max score)
                if y_pred != yi:  # Update weights if prediction is incorrect
                    self.W[yi] += xi
                    self.W[y_pred] -= xi

            # After each epoch, evaluate on validation set if available
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)  # Predict on validation set
                metrics = calculate_metrics(y_val, val_pred, self.num_classes)  # Calculate metrics (accuracy, F1, etc.)
                print_metrics(metrics, epoch=epoch+1, idx2label=self.idx2label)  # Print metrics

                # Check if this epoch gives better validation accuracy
                val_accuracy = metrics['accuracy']  # Assume 'accuracy' is included in the metrics dictionary
                if val_accuracy > best_val_accuracy:  # Update the best epoch if validation accuracy improves
                    best_val_accuracy = val_accuracy
                    best_epoch = epoch

        print(f"Best validation accuracy achieved at epoch {best_epoch+1} with accuracy {best_val_accuracy:.4f}")
    
    def predict(self, X):
        # Predict class labels for input samples
        scores = X @ self.W.T  # Compute scores for each class for each sample
        return np.argmax(scores, axis=1)  # Return the class with the highest score


def plot_perceptron_confusion_matrix(y_true, y_pred, label_names, title, save_path):
    evaluate_and_plot_cm(y_true, y_pred, label_names, title=title, save_path=save_path)


def main():
    train_p = r"C:/Users/24350/env-nlp/emotional-damage/final_submission/train_modi.csv"
    val_p = r"C:/Users/24350/env-nlp/emotional-damage/final_submission/isear-val.csv"
    test_p = r"C:/Users/24350/env-nlp/emotional-damage/final_submission/isear-test.csv"
    LABELS     = {"joy","anger","fear","disgust","sadness","shame","guilt"}
    label_list = sorted(LABELS)
    label2idx  = {l:i for i,l in enumerate(label_list)}
    idx2label  = {i:l for l,i in label2idx.items()}

    # Read data and feature extraction
    tr_s = read_dataset(train_p, LABELS)
    va_s = read_dataset(val_p,   LABELS)
    te_s = read_dataset(test_p,  LABELS)
    X_tr, y_tr, vocab = vectorize_dataset(tr_s)
    X_va, y_va, _     = vectorize_dataset(va_s, vocabulary=vocab)
    X_te, y_te, _     = vectorize_dataset(te_s, vocabulary=vocab)
    y_tr = np.array([label2idx[l] for l in y_tr])
    y_va = np.array([label2idx[l] for l in y_va])
    y_te = np.array([label2idx[l] for l in y_te])

    # Initialize model
    model = MultiPerceptron(input_dim=X_tr.shape[1],
                            num_classes=len(label_list),
                            idx2label=idx2label)

    # Train with shuffling
    print("=== Training: 10 epochs with shuffle ===")
    model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va,
              epochs=10, shuffle_data=True)

    # Final evaluation on test set
    print("\n=== Final Evaluation on Test Set ===")
    y_pred = model.predict(X_te)
    mts = calculate_metrics(y_te, y_pred, model.num_classes)
    print_metrics(mts, epoch=None, idx2label=idx2label)

    # visualized evaluation
    plot_perceptron_confusion_matrix(y_te, y_pred, label_list, title="Perceptron Confusion Matrix", save_path="confusion_matrix_perceptron.png")

if __name__ == "__main__":
    main()
