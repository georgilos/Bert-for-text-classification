import torch
import json
from transformers import BertTokenizer, BertModel
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from data_embeddings_distance_mat import generate_embeddings  # Import your existing generate_embeddings function


def load_model(model_path):
    """
    Load the fine-tuned BERT model.
    """
    model = BertModel.from_pretrained('bert-base-uncased')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model


def load_memory_bank(memory_bank_path):
    """
    Load the memory bank containing the cluster centroids.
    """
    return torch.load(memory_bank_path)


def load_cluster_labels(cluster_labels_path):
    """
    Load the mapping of cluster IDs to labels from the JSON file.
    """
    with open(cluster_labels_path, 'r') as f:
        return json.load(f)


def classify_text(texts, model, tokenizer, memory_bank, cluster_labels):
    """
    Classify texts based on the nearest cluster centroid using cosine similarity.
    """
    # Generate embeddings for the texts (batch processing)
    embeddings = generate_embeddings(texts, tokenizer, model, batch_size=32, use_cls=True)
    embeddings = embeddings.cpu().numpy()  # Ensure embeddings are on CPU for distance calculation

    # Move centroids to CPU and convert to NumPy array
    centroids = np.array([centroid.cpu().numpy() for centroid in memory_bank.values()])

    # Normalize embeddings and centroids
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    # Calculate cosine distances to centroids
    distances = cdist(embeddings, centroids, metric='cosine')

    # Find the nearest cluster for each text
    nearest_cluster_ids = np.argmin(distances, axis=1)

    # Retrieve labels for the nearest clusters
    labels = [cluster_labels.get(str(cluster_id), "Undetermined") for cluster_id in nearest_cluster_ids]

    return labels, nearest_cluster_ids, distances


def main():
    # File paths
    model_path = "./models/fine_tuned_bert.pth"
    memory_bank_path = "./models/final_memory_bank.pt"
    cluster_labels_path = "./models/cluster_labels.json"
    input_texts_path = "./data/unseen_texts.csv"  # File containing unseen texts
    output_predictions_path = "./results/predictions.csv"

    # Load the model, memory bank, and cluster labels
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = load_model(model_path).to(device)
    memory_bank = load_memory_bank(memory_bank_path)
    cluster_labels = load_cluster_labels(cluster_labels_path)

    # Load unseen texts
    unseen_data = pd.read_csv(input_texts_path)
    unseen_data.columns = ['ID', 'TEXT']
    unseen_texts = unseen_data['TEXT'].tolist()

    # Classify texts in batches
    labels, nearest_clusters, distances = classify_text(
        unseen_texts, model, tokenizer, memory_bank, cluster_labels
    )

    # Prepare results for saving
    predictions = [
        {"Text": text, "Label": label, "Cluster": cluster, "Distances": dist.tolist()}
        for text, label, cluster, dist in zip(unseen_texts, labels, nearest_clusters, distances)
    ]

    # Save the predictions to a CSV file
    output_df = pd.DataFrame(predictions)
    output_df.to_csv(output_predictions_path, index=False)
    print(f"Predictions saved to {output_predictions_path}")


if __name__ == "__main__":
    main()
