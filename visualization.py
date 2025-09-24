import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from preprocessing import read_dataset

SAVE_DIR = r"C:\Users\24350\env-nlp\emotional-damage\report figures"

# Figure 1: visualize the emotion distribution
def visualize_emotion_distribution(samples):

    emotion_counts = Counter([sample.emotion for sample in samples])
    plt.figure(figsize=(12, 8))
    bars = plt.bar(emotion_counts.keys(), emotion_counts.values(), color='#666666') 

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.xlabel('Emotion labels', fontsize=20)  
    plt.ylabel('Frequency', fontsize=20)  
    plt.title('Frequency of Emotion labels', fontsize=22) 
    plt.xticks(rotation=45, ha='right', fontsize=16)  
    plt.yticks(fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, 'emotion_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"The emotion distribution figure is saved as '{save_path}'")
    plt.close() 
    
    return emotion_counts  

# Figure 2: visualize the overall accuracy and F1 for 3 models
def visualize_model_comparison(file_path):
    df = pd.read_excel(file_path, header=None) 
    models = df.iloc[0, 1:].values.tolist()  
    accuracy = df.iloc[1, 1:].astype(float).values  
    f1_score = df.iloc[2, 1:].astype(float).values  
    x = range(len(models))  
    fig, ax = plt.subplots(figsize=(12, 8)) 
    bar_width = 0.35  
    
    bars1 = ax.bar(x, accuracy, bar_width, label='Overall Accuracy', color='#444444')  
    bars2 = ax.bar([p + bar_width for p in x], f1_score, bar_width, label='Overall F1', color='#888888')  
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # dummy comparison
    dummy_accuracy = 0.1315
    dummy_f1 = 0.0000
    ax.axhline(y=dummy_accuracy, color='red', linestyle='--', linewidth=2, label=f'Dummy Accuracy: {dummy_accuracy:.3f}')
    ax.axhline(y=dummy_f1, color='red', linestyle='-', linewidth=2, label=f'Dummy F1: {dummy_f1:.3f}')
    
    ax.set_xlabel('Model', fontsize=20)  
    ax.set_ylabel('Scores', fontsize=20)  
    ax.set_title('Overall Accuracy and F1 Score for Different Models', fontsize=22)  
    ax.set_xticks([p + bar_width / 2 for p in x])
    ax.set_xticklabels(models, fontsize=16)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, 'model_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"The model comparison figure is saved as '{save_path}'")
    plt.close()  

# Figure 3: visualize the F1 scores for Perceptron, FFNN, and Naïve Bayes models across different labels
def visualize_f1_comparison(file_path):
    df = pd.read_excel(file_path, header=None)

    labels = df.iloc[1:8, 0].tolist()  
    perceptron_f1 = df.iloc[1:8, 1].astype(float).values  
    ffnn_f1 = df.iloc[1:8, 2].astype(float).values  
    naive_f1 = df.iloc[1:8, 3].astype(float).values  

    x = range(len(labels))  

    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.25  

    bars1 = ax.bar(x, perceptron_f1, bar_width, label='Perceptron', color='#ff7f0e')  
    bars2 = ax.bar([p + bar_width for p in x], ffnn_f1, bar_width, label='FFNN', color='#2ca02c')  
    bars3 = ax.bar([p + bar_width*2 for p in x], naive_f1, bar_width, label='Naïve Bayes', color='#1f77b4')  

    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    for bar in bars3:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Label', fontsize=20)
    ax.set_ylabel('F1 Score', fontsize=20)
    ax.set_title('F1 Score of Each Model on Each Label', fontsize=22)
    ax.set_xticks([p + bar_width for p in x])
    ax.set_xticklabels(labels, fontsize=12, rotation=45, ha='right')

    ax.legend()

    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, 'f1_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"The F1 comparison figure is saved as '{save_path}'")
    plt.close()

if __name__ == '__main__':
    # Figure 1: visualize the emotion distribution
    file_path = os.path.join("C:/Users/24350/env-nlp/emotional-damage/final_submission/train_modi.csv")
    allowed_labels = {'joy', 'anger', 'fear', 'disgust', 'sadness', 'shame', 'guilt'}
    data = read_dataset(file_path, allowed_labels)

    if data:
        print(f'Loaded {len(data)} samples')  
        emotion_distribution = visualize_emotion_distribution(data)  
        print("Emotion distribution:", emotion_distribution)  

    # Figure 2: visualize the overall accuracy and F1 for 3 models
    excel_file_path = r'C:/Users/24350/env-nlp/emotional-damage/final_submission/final/Overall_accuracy_and_F1_score.xlsx'  
    visualize_model_comparison(excel_file_path)  

    # Figure 3: visualize the F1 comparison for different models
    f1_excel_file_path = r'C:/Users/24350/env-nlp/emotional-damage/final_submission/final/models_on_7.xlsx'  
    visualize_f1_comparison(f1_excel_file_path)  
