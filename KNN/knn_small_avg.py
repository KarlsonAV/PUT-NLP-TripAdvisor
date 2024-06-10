import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import shuffle

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train = train_data['text'].tolist()
y_train = train_data['label'].tolist()

X_test = test_data['text'].tolist()
y_test = test_data['label'].tolist()

# Load the small embeddings model
model_name_small = 'jinaai/jina-embeddings-v2-small-en'
tokenizer_small = AutoTokenizer.from_pretrained(model_name_small)
model_small = AutoModel.from_pretrained(model_name_small).to(device)

# Function to compute embeddings
def get_embeddings(texts, tokenizer, model, device, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)
    return embeddings

# Compute embeddings for training data
X_train_embeddings_small = get_embeddings(X_train, tokenizer_small, model_small, device)

# Function to compute average embeddings for each class
def average_embeddings(X, y):
    unique_labels = np.unique(y)
    avg_embeddings = []
    for label in unique_labels:
        label_embeddings = X[np.array(y) == label]
        avg_embedding = np.mean(label_embeddings, axis=0)
        avg_embeddings.append(avg_embedding)
    return np.array(avg_embeddings), unique_labels

# Compute average embeddings for training data
avg_embeddings_small, labels_small = average_embeddings(X_train_embeddings_small, y_train)

# Train KNN classifier on average embeddings
knn_small = KNeighborsClassifier(n_neighbors=1)
knn_small.fit(avg_embeddings_small, labels_small)

# Compute embeddings for test data
X_test_embeddings_small = get_embeddings(X_test, tokenizer_small, model_small, device)

# Make predictions on test data
y_pred_small = knn_small.predict(X_test_embeddings_small)

# Print classification report
print("Small Embeddings:")
print(classification_report(y_test, y_pred_small))

# Save the model
with open('knn_small_avg.pkl', 'wb') as f:
    pickle.dump(knn_small, f)

# Compute confusion matrix
conf_matrix_small = confusion_matrix(y_test, y_pred_small)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_small, annot=True, fmt="d", cmap="Blues")
plt.title("Small Embeddings")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
