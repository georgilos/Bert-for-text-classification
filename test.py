import os
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from collections import Counter
from scipy.spatial.distance import cdist
from constrained_clustering import constrained_dbscan_with_constraints, merge_small_clusters, initialize_memory_bank
from uncertain_pairs import select_uncertain_pairs, annotate_and_update_constraints
from data_embeddings_distance_mat import generate_embeddings
from hybrid_loss_training import calculate_contrastive_loss, calculate_support_pair_loss


def iterative_training(all_texts, max_iterations=4, margin=1.0, temperature=0.5, lambda_t=1.0):
    """
    Perform iterative training with dynamic eps and min_samples selection.
    """
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.train()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Step 1: Generate Initial Embeddings
    print("Generating initial embeddings...")
    sampled_embeddings = generate_embeddings(all_texts, tokenizer, model, batch_size=16, use_cls=True)
    distance_matrix = cdist(sampled_embeddings.cpu().numpy(), sampled_embeddings.cpu().numpy(), metric='cosine')

    # Calculate and display mean pairwise distance
    mean_distance = np.mean(distance_matrix)
    print(f"Mean Pairwise Distance: {mean_distance:.4f}")

    # Dynamic parameter adjustment for initial clustering
    while True:
        eps = float(input(
            f"Enter the eps value for initial clustering (default suggestion: {mean_distance * 0.5:.4f}): ") or mean_distance * 0.5)
        min_samples = int(input(f"Enter the min_samples value for initial clustering (default suggestion: 2): ") or 2)

        # Perform initial clustering
        adjusted_labels, cannot_link_dict = constrained_dbscan_with_constraints(distance_matrix, eps, min_samples, must_link=[],
                                                              cannot_link=[])
        print("Now merging clusters with instance<min_samples")
        adjusted_labels = merge_small_clusters(distance_matrix, adjusted_labels, cannot_link_dict, min_samples)
        # Print clustering results
        unique_clusters = np.unique(adjusted_labels)
        print(f"\nInitial Clustering Results (eps={eps}, min_samples={min_samples}):")
        print(f"Clusters Found: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)} (excluding noise)")
        # Count the instances in each cluster
        cluster_counts = Counter(adjusted_labels)
        for cluster_id, count in cluster_counts.items():
            print(f"Cluster {cluster_id}: {count} instances")

        # Confirm or adjust parameters
        user_choice = input("Are you satisfied with the initial clustering? (y/n): ").strip().lower()
        if user_choice == "y":
            break
        else:
            print("Reclustering with different parameters...")

    # Initialize memory bank
    memory_bank = initialize_memory_bank(sampled_embeddings, torch.tensor(adjusted_labels, dtype=torch.int64))

    # Initialize constraints
    must_link_pairs, cannot_link_pairs = [], []

    # Iterative process
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}:")

        # Step 2: Select Uncertain Pairs and Annotate
        uncertain_positive_pairs, uncertain_negative_pairs = select_uncertain_pairs(distance_matrix, adjusted_labels)
        must_link_pairs, cannot_link_pairs = annotate_and_update_constraints(
            uncertain_positive_pairs, uncertain_negative_pairs, all_texts, must_link_pairs, cannot_link_pairs
        )

        # Step 3: Fine-Tune Model
        print("Fine-tuning model...")
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        for i in range(0, len(all_texts), 32):  # Batch size = 32
            batch_texts = all_texts[i:i + 32]
            batch_indices = list(range(i, min(i + 32, len(all_texts))))
            batch_index_set = set(batch_indices)

            # Tokenize and generate embeddings
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token embeddings

            # Filter must-link and cannot-link pairs
            batch_must_link_pairs = [(a, b) for (a, b) in must_link_pairs if
                                     a in batch_index_set and b in batch_index_set]
            batch_cannot_link_pairs = [(a, b) for (a, b) in cannot_link_pairs if
                                       a in batch_index_set and b in batch_index_set]

            # Create a mapping from global to batch-local indices
            index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(batch_indices)}

            # Convert global indices in pairs to batch-local indices
            batch_must_link_pairs = [
                (index_map[a], index_map[b]) for (a, b) in batch_must_link_pairs
            ]
            batch_cannot_link_pairs = [
                (index_map[a], index_map[b]) for (a, b) in batch_cannot_link_pairs
            ]

            # Compute hybrid loss
            contrastive_loss = calculate_contrastive_loss(memory_bank, embeddings,
                                                          torch.tensor(adjusted_labels)[batch_indices], temperature)
            support_pair_loss = calculate_support_pair_loss(embeddings, batch_must_link_pairs, batch_cannot_link_pairs,
                                                            margin)
            combined_loss = contrastive_loss + lambda_t * support_pair_loss

            # Update model
            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()

        # Step 4: Recompute Embeddings, Clusters, and Memory Bank
        print("Recomputing embeddings and clustering...")
        updated_embeddings = generate_embeddings(all_texts, tokenizer, model, batch_size=16, use_cls=True)
        distance_matrix = cdist(updated_embeddings.cpu().numpy(), updated_embeddings.cpu().numpy(), metric='cosine')

        # Calculate and display mean pairwise distance
        mean_distance = np.mean(distance_matrix)
        print(f"Mean Pairwise Distance (Iteration {iteration + 1}): {mean_distance:.4f}")

        # Dynamic parameter adjustment for clustering
        while True:
            eps = float(input(
                f"Enter the eps value for clustering (default suggestion: {mean_distance * 0.5:.4f}): ") or mean_distance * 0.5)
            min_samples = int(input(f"Enter the min_samples value for clustering (default suggestion: 2): ") or 2)

            # Perform clustering
            adjusted_labels, cannot_link_dict = constrained_dbscan_with_constraints(distance_matrix, eps, min_samples, must_link_pairs,
                                                                  cannot_link_pairs)
            adjusted_labels = merge_small_clusters(distance_matrix, adjusted_labels, cannot_link_dict, min_samples)
            # Print clustering results
            unique_clusters = np.unique(adjusted_labels)
            print(f"\nClustering Results (eps={eps}, min_samples={min_samples}):")
            print(f"Clusters Found: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)} (excluding noise)")
            # Count the instances in each cluster
            cluster_counts = Counter(adjusted_labels)
            for cluster_id, count in cluster_counts.items():
                print(f"Cluster {cluster_id}: {count} instances")

            # Confirm or adjust parameters
            user_choice = input("Are you satisfied with the clustering? (y/n): ").strip().lower()
            if user_choice == "y":
                break
            else:
                print("Reclustering with different parameters...")

        memory_bank = initialize_memory_bank(updated_embeddings, torch.tensor(adjusted_labels, dtype=torch.int64))

    # View cluster summaries
    print("\nFinal Cluster Summaries:")
    cluster_counts = Counter(adjusted_labels)
    for cluster_id, count in cluster_counts.items():
        if cluster_id == -1:
            print(f"Cluster {cluster_id} (Noise): {count} instances")
        else:
            print(f"Cluster {cluster_id}: {count} instances")

    # Save the cluster assignments and centroids
    cluster_labels = {}  # Dictionary to store cluster IDs and their labels
    print("\nLabel clusters as 'Hate' or 'Not Hate':")
    for cluster_id in cluster_counts.keys():
        if cluster_id == -1:
            cluster_labels[cluster_id] = "Noise"
        else:
            print(f"\nCluster {cluster_id}:")
            sample_texts = [all_texts[i] for i in range(len(all_texts)) if adjusted_labels[i] == cluster_id][:5]
            print("Sample Texts:")
            for text in sample_texts:
                print(f"- {text}")
            label = input(f"Label for Cluster {cluster_id} ('Hate' or 'Not Hate'): ").strip()
            cluster_labels[cluster_id] = label

    # Save cluster centroids and labels for inference
    torch.save(memory_bank, "./models/final_memory_bank.pt")
    # Convert numpy.int64 keys to int
    cluster_labels = {int(k): v for k, v in cluster_labels.items()}
    with open("./models/cluster_labels.json", "w") as f:
        import json
        json.dump(cluster_labels, f)

    print("Final memory bank and cluster labels saved!")
    print(cluster_labels)
    # Save the fine-tuned model
    save_path = "./models/fine_tuned_bert.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuned model saved to {save_path}")


def main():
    data_path = "data/unlabeled_data/cleaned_texts_unlabeled.csv"  # Path to the CSV file
    sampled_data = pd.read_csv(data_path, header=None)  # Load the CSV file (no headers)
    sampled_data.columns = ['ID', 'TEXT']  # Add column names to the CSV

    # Clean and validate TEXT column
    sampled_data['TEXT'] = sampled_data['TEXT'].fillna('').astype(str).str.replace(r'[\ufe0f\x0f]', '', regex=True)
    sampled_data = sampled_data[sampled_data['TEXT'].str.strip() != '']

    # Sample and prepare the data
    sampled_data = sampled_data.sample(n=300, random_state=55)  # Randomly sample 100 rows
    all_texts = sampled_data['TEXT'].tolist()

    # Run iterative training
    iterative_training(all_texts, max_iterations=4)


if __name__ == "__main__":
    main()
