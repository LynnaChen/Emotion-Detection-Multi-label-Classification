from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def calculate_metrics(y_true, y_pred, num_classes):
    # Calculate accuracy
    accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
    
    # Initialize metrics for each class
    precision = [0.0] * num_classes
    recall = [0.0] * num_classes
    f1 = [0.0] * num_classes
    
    # Calculate metrics for each class
    for label in range(num_classes):
        # Calculate true positives, false positives, false negatives
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        
        # Calculate precision
        precision[label] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Calculate recall
        recall[label] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Calculate F1 score
        f1[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label]) if (precision[label] + recall[label]) > 0 else 0.0
    
    # Calculate macro-F1
    macro_f1 = sum(f1) / num_classes
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': macro_f1,
        'predictions': y_pred
    }

def print_metrics(metrics, epoch=None, idx2label=None):
    if epoch is not None:
        print(f"\nEpoch {epoch} - Validation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    
    if idx2label is not None:
        print("\nPer-class metrics:")
        for i in range(len(idx2label)):
            label_name = idx2label[i]
            print(f"{label_name}: Precision={metrics['precision'][i]:.4f}, "
                  f"Recall={metrics['recall'][i]:.4f}, F1={metrics['f1'][i]:.4f}")
    print("-------------------------------------------------------")

def print_evaluation(y_true, y_pred, idx2label=None):
    num_classes = len(set(y_true) | set(y_pred))
    metrics = calculate_metrics(y_true, y_pred, num_classes)
    print_metrics(metrics, idx2label=idx2label)

def evaluate_and_plot_cm(y_true, y_pred, label_names, title="Confusion Matrix", save_path=None):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print()
    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(xticks_rotation=45, cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()
