import torch
import json
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

# Load the fine-tuned BERT model
model_path = "./models/fine_tuned_bert.pth"
memory_bank_path = "./models/final_memory_bank.pt"
cluster_labels_path = "./models/cluster_labels.json"

# Load model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.load_state_dict(torch.load(model_path))
model.eval()

# Load memory bank and cluster labels
memory_bank = torch.load(memory_bank_path)
with open(cluster_labels_path, "r") as f:
    cluster_labels = json.load(f)


# Function to embed a new sentence
def embed_sentence(sentence, tokenizer, model, use_cls=True):
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    if use_cls:
        embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    else:
        embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return F.normalize(embedding, p=2, dim=1)  # Normalize


# Predict the label of a new sentence
def predict_label(sentence, tokenizer, model, memory_bank, cluster_labels):
    embedding = embed_sentence(sentence, tokenizer, model)
    max_similarity = -float('inf')
    predicted_cluster = None

    # Compare with all centroids in the memory bank
    for cluster_id, centroid in memory_bank.items():
        similarity = torch.matmul(embedding, centroid.unsqueeze(1)).item()
        if similarity > max_similarity:
            max_similarity = similarity
            predicted_cluster = cluster_id

    # Get the label of the closest cluster
    return cluster_labels.get(str(predicted_cluster), "Unknown")


# Test the function
if __name__ == "__main__":
    test_sentence = input("Enter a sentence to classify: ").strip()
    label = predict_label(test_sentence, tokenizer, model, memory_bank, cluster_labels)
    print(f"Predicted Label: {label}")
