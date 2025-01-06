import os
import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import numpy as np
import pandas as pd


def calculate_contrastive_loss(memory_bank, embeddings, cluster_labels, temperature=0.5):
    """
    Calculate the contrastive loss (L_c) based on instance-to-centroid contrastive loss.
    """
    # Filter out noise points (cluster label == -1)
    valid_indices = cluster_labels != -1
    valid_embeddings = embeddings[valid_indices]
    valid_embeddings = F.normalize(valid_embeddings, p=2, dim=1)
    valid_labels = cluster_labels[valid_indices]

    # Prepare the centroids for all valid points
    centroids = torch.stack([memory_bank[int(label.item())] for label in valid_labels])
    centroids = F.normalize(centroids, p=2, dim=1)

    # Calculate logits: instance-to-centroid similarity
    batch_size = 8
    logits = []
    for i in range(0, len(valid_embeddings), batch_size):
        batch_embeddings = valid_embeddings[i:i + batch_size]
        batch_logits = torch.mm(batch_embeddings, centroids.T) / temperature
        logits.append(batch_logits)
    logits = torch.cat(logits, dim=0)

    # Create the labels for the contrastive loss
    labels = torch.arange(len(valid_embeddings)).to(logits.device)

    # Calculate contrastive loss using CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, labels)

    return loss


def calculate_support_pair_loss(embeddings, must_link_pairs, cannot_link_pairs, margin=1.0):
    """
    Calculate the Support Pair Constraints Loss (L_t) based on must-link and cannot-link constraints.
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Precompute pairwise distances
    pairwise_distances = 1 - torch.matmul(embeddings, embeddings.T)

    triplet_losses = []

    for anchor_idx in range(embeddings.size(0)):
        positives = [j for (a, j) in must_link_pairs if a == anchor_idx] + \
                    [a for (a, j) in must_link_pairs if j == anchor_idx]
        negatives = [j for (a, j) in cannot_link_pairs if a == anchor_idx] + \
                    [a for (a, j) in cannot_link_pairs if j == anchor_idx]

        if not positives or not negatives:
            continue

        # Find the hardest positive and hardest negative
        hardest_positive_idx = torch.argmax(torch.tensor([pairwise_distances[anchor_idx, p] for p in positives]))
        hardest_positive = positives[hardest_positive_idx]

        hardest_negative_idx = torch.argmin(torch.tensor([pairwise_distances[anchor_idx, n] for n in negatives]))
        hardest_negative = negatives[hardest_negative_idx]

        # Compute triplet loss
        positive_distance = pairwise_distances[anchor_idx, hardest_positive]
        negative_distance = pairwise_distances[anchor_idx, hardest_negative]
        triplet_loss = F.relu(positive_distance - negative_distance + margin)
        triplet_losses.append(triplet_loss)

    # Average triplet losses
    if triplet_losses:
        support_pair_loss = torch.stack(triplet_losses).mean()
    else:
        support_pair_loss = torch.tensor(0.0, requires_grad=True)

    return support_pair_loss


def train_model(memory_bank, adjusted_labels, must_link_pairs, cannot_link_pairs, all_texts):
    """
    Train the model using a hybrid loss function.
    """
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.train()

    # Move model and data to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    adjusted_labels = adjusted_labels.to(device)
    memory_bank = {k: v.to(device) for k, v in memory_bank.items()}

    # Hyperparameters
    batch_size = 32
    lambda_t = 1.0  # Weight for support pair loss
    margin = 0.1  # Margin for triplet loss
    temperature = 0.5  # Temperature for contrastive loss
    num_epochs = 3

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # Training loop
    for epoch in range(num_epochs):
        for i in range(0, len(all_texts), batch_size):
            # Prepare batch data
            batch_texts = all_texts[i:i + batch_size]
            batch_indices = list(range(i, min(i + batch_size, len(all_texts))))
            batch_index_set = set(batch_indices)

            # Tokenize and generate embeddings for the batch
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token embeddings

            # Filter must-link and cannot-link pairs for the current batch
            batch_must_link_pairs = [
                (a, b) for (a, b) in must_link_pairs if a in batch_index_set and b in batch_index_set
            ]
            batch_cannot_link_pairs = [
                (a, b) for (a, b) in cannot_link_pairs if a in batch_index_set and b in batch_index_set
            ]

            # Create a mapping from global to batch-local indices
            index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(batch_indices)}

            # Convert global indices in pairs to batch-local indices
            batch_must_link_pairs = [
                (index_map[a], index_map[b]) for (a, b) in batch_must_link_pairs
            ]
            batch_cannot_link_pairs = [
                (index_map[a], index_map[b]) for (a, b) in batch_cannot_link_pairs
            ]

            # Compute losses
            contrastive_loss = calculate_contrastive_loss(memory_bank, embeddings, adjusted_labels[batch_indices], temperature=temperature)
            support_pair_loss = calculate_support_pair_loss(embeddings, batch_must_link_pairs, batch_cannot_link_pairs, margin=margin)

            # Combine losses
            combined_loss = contrastive_loss + lambda_t * support_pair_loss

            # Backpropagation
            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()

            # Debugging: Monitor losses
            print(f"Epoch {epoch + 1}, Batch {i // batch_size + 1}: "
                  f"Contrastive Loss = {contrastive_loss.item():.4f}, "
                  f"Support Pair Loss = {support_pair_loss.item():.4f}, "
                  f"Combined Loss = {combined_loss.item():.4f}")

    # Save the fine-tuned model
    save_path = "./models/fine_tuned_bert.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuned BERT model saved to {save_path}")


def main():
    # Load data
    memory_bank = torch.load("memory_bank.pt")
    adjusted_labels = torch.tensor(np.load("adjusted_labels.npy"), dtype=torch.int64)
    sampled_data = pd.read_csv("sampled_data.csv")
    all_texts = sampled_data['TEXT'].tolist()

    # Load constraints
    must_link_pairs = np.load("must_link_pairs.npy", allow_pickle=True).tolist()
    cannot_link_pairs = np.load("cannot_link_pairs.npy", allow_pickle=True).tolist()

    # Train the model
    train_model(memory_bank, adjusted_labels, must_link_pairs, cannot_link_pairs, all_texts)


if __name__ == "__main__":
    main()
