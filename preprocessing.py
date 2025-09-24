import csv
import re

class Text:
    def __init__(self, raw_text: str):
        self.text = raw_text
        self.tokens = self.tokenize(raw_text)

    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        return text.split()

class Emotions:  # Emotion data class
    def __init__(self, emotion, text):
        self.emotion = emotion
        self.text = text
        self.tokens = Text(self.text).tokens
        self.features = self.tokens

def read_dataset(file_path, allowed_labels):
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get('label') or '').strip().lower()
            text = (row.get('text') or '').strip()
            if label in allowed_labels and text:
                samples.append(Emotions(label, text))
    return samples

def preprocess_csv(input_csv, output_csv):
    sample_tokens = []
    with open(input_csv, 'r', encoding='utf-8') as f, open(output_csv, 'w', encoding='utf-8', newline='') as out_f:
        reader = csv.reader(f)
        writer = csv.writer(out_f)
        for idx, row in enumerate(reader):
            if not row or len(row) < 2:  # Handle empty or malformed rows
                continue
            label, text = row[0], row[1]
            text_obj = Text(text)
            processed = ' '.join(text_obj.tokens)
            writer.writerow([label, processed])
            if len(sample_tokens) < 20:
                sample_tokens.append((text, text_obj.tokens))
    return sample_tokens

if __name__ == '__main__':
    input_csv = r'C:/Users/24350/env-nlp/emotional-damage/data/isear/isear/train_modi.csv'
    output_csv = r'C:/Users/24350/env-nlp/emotional-damage/final submission/train_modi.csv'
    samples = preprocess_csv(input_csv, output_csv)
    print('Preprocessing done, results saved to', output_csv)
    print("\nResults of the first 20 samples:")
    for idx, (raw, tokens) in enumerate(samples):
        print(f"\nSample {idx+1}:")
        print(f"Original text: {raw}")
        print(f"Tokenization results: {tokens}")
