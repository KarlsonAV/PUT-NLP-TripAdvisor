import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the dataset
test_data = pd.read_csv('test.csv')

# Inspect the dataset to see the column names
print(test_data.columns)

# Assuming the column names are 'review' for the text and 'label' for the true labels
true_label_column = 'label'  # Update this if your column name is different
review_column = 'text'     # Update this if your column name is different

# Load the models from Hugging Face and move them to the GPU
model_names = [
    "deptage/distilbert1",
    "deptage/distilbert2",
    "deptage/distilbert3",
    "deptage/distilbert4",
    "deptage/distilbert5",
]

models = [DistilBertForSequenceClassification.from_pretrained(name).to(device) for name in model_names]
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def predict_label(review_text):
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to the GPU
    probabilities = []

    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prob = torch.softmax(logits, dim=1)[0][1].item()  # Get probability of class 1
            probabilities.append(prob)
    
    # Get the label with the highest probability
    predicted_label = probabilities.index(max(probabilities)) + 1
    return predicted_label

# Evaluate the models on the test dataset
true_labels = test_data[true_label_column].tolist()
predicted_labels = [predict_label(review) for review in test_data[review_column]]

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=[1, 2, 3, 4, 5])

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
