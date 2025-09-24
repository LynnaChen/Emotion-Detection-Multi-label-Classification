# Emotional Damage

All files about the project are stored under `final_submission_code`.

## Description
This project focuses on emotion classification using the ISEAR (International Survey on Emotion Antecedents and Reactions) dataset, which contains seven emotion categories: joy, anger, fear, disgust, sadness, shame, and guilt. We began by implementing a multi-class perceptron. Based on the results, we introduced more advanced methods—Naive Bayes with TF-IDF with n-gram feature extraction, and a Feedforward Neural Network (FFNN) with TF-IDF feature extraction. We evaluated models using evaluation methods with accuracy, macro F1-score, and confusion matrix.

## File Structure

- **`preprocessing.py`**
  Provides text preprocessing pipeline including lowercase conversion, punctuation removal, tokenization, and CSV data loading functionality.

- **`evaluation_F1_CM.py`**  
  Evaluation method using accuracy, precision, recall, F1 from scratch, and confusion matrix is added to give more details.

- **`dummy.py`**  
  Uses a majority vote strategy: find the most frequent emotion label in the training set, then predicts this same label for all validation samples to establish a baseline performance for the task.

- **`BOW+perceptron.py`**  
  Combines bag-of-words feature extraction with a multi-class perceptron classifier.

- **`n-gram+naive.py`**  
  A simple Naive Bayes classifier using TF-IDF-weighted n-gram features for emotion classification.

- **`tf-idf+FFNN.py`**  
  A feedforward neural network (FFNN) model using TF-IDF features for emotion classification.

- **`visualization.py`**  
  Generates bar charts to visualize model performance and label distribution.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv <name-of-your-virtual-environment>
   
2. Clone the repository:
   ```bash
   git clone https://github.tik.uni-stuttgart.de/st195343/emotional-damage.git
   cd emotional-damage/final_submission_code

3. Install required packages:
   ```bash
   pip install numpy pandas scikit-learn torch matplotlib

3. Run the code: Once the setup is complete, you can run any of the scripts. For example, to run the ```bash evaluation_F1_CM.py script, use:
   ```bash
   python evaluation_F1_CM.py
