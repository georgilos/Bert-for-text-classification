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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_embeddings(embeddings, labels, iteration, stage, title="Embedding Visualization"):
    """
    Visualize embeddings using t-SNE and save the plot.

    Parameters:
    - embeddings: numpy array of embeddings.
    - labels: cluster labels corresponding to embeddings.
    - iteration: current iteration number.
    - stage: 'before' or 'after' training.
    - title: title of the plot.
    """
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="tab10", s=5)
    plt.colorbar(scatter)
    plt.title(f"{title} (Iteration {iteration} - {stage} Training)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(f"embeddings_iteration_{iteration}_{stage}.png")
    plt.close()


def iterative_training(all_texts, max_iterations=4, margin=1.0, temperature=0.5, lambda_t=1.0):
    """
    Perform iterative training with monitoring and visualization of embeddings.
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
    embeddings_np = sampled_embeddings.cpu().numpy()

    # Visualize embeddings before any training
    labels_placeholder = np.zeros(len(all_texts))  # Since no labels yet, use zeros
    plot_embeddings(embeddings_np, labels_placeholder, iteration=0, stage='before', title="Initial Embeddings")

    distance_matrix = cdist(embeddings_np, embeddings_np, metric='cosine')

    # Calculate and display mean pairwise distance
    mean_distance = np.mean(distance_matrix)
    print(f"Mean Pairwise Distance: {mean_distance:.4f}")

    # Dynamic parameter adjustment for initial clustering
    while True:
        eps = float(input(
            f"Enter the eps value for initial clustering (default suggestion: {mean_distance * 0.5:.4f}): ") or mean_distance * 0.5)
        min_samples = int(input(f"Enter the min_samples value for initial clustering (default suggestion: 2): ") or 2)

        # Perform initial clustering
        adjusted_labels, cannot_link_dict = constrained_dbscan_with_constraints(
            distance_matrix, eps, min_samples, must_link=[], cannot_link=[]
        )
        print("Now merging clusters with instance < min_samples")
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

    # Visualize embeddings with initial clustering
    plot_embeddings(embeddings_np, adjusted_labels, iteration=0, stage='after',
                    title="Embeddings After Initial Clustering")

    # Initialize memory bank
    memory_bank = initialize_memory_bank(sampled_embeddings, torch.tensor(adjusted_labels, dtype=torch.int64))

    # Initialize constraints
    must_link_pairs, cannot_link_pairs = [], []

    # Track inclusion of constraints
    must_link_inclusion = {pair: 0 for pair in must_link_pairs}
    cannot_link_inclusion = {pair: 0 for pair in cannot_link_pairs}

    # Iterative process
    for iteration in range(1, max_iterations + 1):
        print(f"\nIteration {iteration}:")

        # Step 2: Select Uncertain Pairs and Annotate
        uncertain_positive_pairs, uncertain_negative_pairs = select_uncertain_pairs(distance_matrix, adjusted_labels)
        must_link_pairs, cannot_link_pairs = annotate_and_update_constraints(
            uncertain_positive_pairs, uncertain_negative_pairs, all_texts, must_link_pairs, cannot_link_pairs
        )

        # Step 2.5: Update tracking for new constraints
        for pair in must_link_pairs:
            if pair not in must_link_inclusion:
                must_link_inclusion[pair] = 0
        for pair in cannot_link_pairs:
            if pair not in cannot_link_inclusion:
                cannot_link_inclusion[pair] = 0

        # Step 3: Fine-Tune Model
        print("Fine-tuning model...")
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        model.train()  # Ensure model is in training mode
        for epoch in range(1):  # Number of epochs per iteration
            for i in range(0, len(all_texts), 32):  # Batch size = 32
                # Create initial batch indices
                batch_indices = list(range(i, min(i + 32, len(all_texts))))

                # Prioritize annotated pairs
                prioritized_indices = set()
                for a, b in must_link_pairs + cannot_link_pairs:
                    prioritized_indices.update([a, b])  # Include both ends of each constraint

                # Merge prioritized indices with the current batch
                batch_indices = list(prioritized_indices.union(batch_indices))
                batch_indices = batch_indices[:32]  # Ensure batch size limit
                batch_index_set = set(batch_indices)

                # Tokenize and generate embeddings for the batch
                batch_texts = [all_texts[idx] for idx in batch_indices]
                inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
                inputs = {key: val.to(device) for key, val in inputs.items()}
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token embeddings

                # Map indices and filter pairs for the batch
                index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(batch_indices)}
                batch_must_link_pairs = [
                    (index_map[a], index_map[b]) for (a, b) in must_link_pairs if
                    a in batch_index_set and b in batch_index_set
                ]
                batch_cannot_link_pairs = [
                    (index_map[a], index_map[b]) for (a, b) in cannot_link_pairs if
                    a in batch_index_set and b in batch_index_set
                ]

                # Add this snippet for monitoring
                print(f"Batch Indices: {batch_indices}")
                print(f"Batch Must-Link Pairs: {batch_must_link_pairs}")
                print(f"Batch Cannot-Link Pairs: {batch_cannot_link_pairs}")

                # Compute hybrid loss
                contrastive_loss = calculate_contrastive_loss(memory_bank, embeddings,
                                                              torch.tensor(adjusted_labels)[batch_indices], temperature)

                # Compute pairwise distances for the batch
                pairwise_distances = torch.cdist(embeddings, embeddings, p=2)

                # Add this snippet before computing support pair loss
                if batch_must_link_pairs:
                    for a, b in batch_must_link_pairs:
                        print(f"Must-Link Pair ({a}, {b}) Distance: {pairwise_distances[a, b]}")

                if batch_cannot_link_pairs:
                    for a, b in batch_cannot_link_pairs:
                        print(f"Cannot-Link Pair ({a}, {b}) Distance: {pairwise_distances[a, b]}")

                # Compute support pair loss
                support_pair_loss = calculate_support_pair_loss(embeddings, batch_must_link_pairs,
                                                                batch_cannot_link_pairs, margin)
                combined_loss = contrastive_loss + lambda_t * support_pair_loss

                # Update model
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()

                # Monitor losses
                print(f"Iteration {iteration}, Batch {i // 32 + 1}: "
                      f"Contrastive Loss = {contrastive_loss.item():.4f}, "
                      f"Support Pair Loss = {support_pair_loss.item():.4f}, "
                      f"Combined Loss = {combined_loss.item():.4f}")

                # Update inclusion counts for constraints
                for a_local, b_local in batch_must_link_pairs:
                    a_global = batch_indices[a_local]  # Get global index from local
                    b_global = batch_indices[b_local]  # Get global index from local
                    must_link_inclusion[(a_global, b_global)] += 1
                for a_local, b_local in batch_cannot_link_pairs:
                    a_global = batch_indices[a_local]  # Get global index from local
                    b_global = batch_indices[b_local]  # Get global index from local
                    cannot_link_inclusion[(a_global, b_global)] += 1

        # Step 4: Recompute Embeddings, Clusters, and Memory Bank
        print("Recomputing embeddings and clustering...")
        updated_embeddings = generate_embeddings(all_texts, tokenizer, model, batch_size=16, use_cls=True)
        embeddings_np = updated_embeddings.cpu().numpy()

        # Visualize embeddings before clustering
        plot_embeddings(embeddings_np, adjusted_labels, iteration=iteration, stage='before',
                        title="Embeddings Before Clustering")

        distance_matrix = cdist(embeddings_np, embeddings_np, metric='cosine')

        # Calculate and display mean pairwise distance
        mean_distance = np.mean(distance_matrix)
        print(f"Mean Pairwise Distance (Iteration {iteration}): {mean_distance:.4f}")

        # Dynamic parameter adjustment for clustering
        while True:
            eps = float(input(
                f"Enter the eps value for clustering (default suggestion: {mean_distance * 0.5:.4f}): ") or mean_distance * 0.5)
            min_samples = int(input(f"Enter the min_samples value for clustering (default suggestion: 2): ") or 2)

            # Perform clustering
            adjusted_labels, cannot_link_dict = constrained_dbscan_with_constraints(
                distance_matrix, eps, min_samples, must_link_pairs, cannot_link_pairs
            )
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

        # Visualize embeddings after clustering
        plot_embeddings(embeddings_np, adjusted_labels, iteration=iteration, stage='after',
                        title="Embeddings After Clustering")

        memory_bank = initialize_memory_bank(updated_embeddings, torch.tensor(adjusted_labels, dtype=torch.int64))

    # View cluster summaries
    print("\nFinal Cluster Summaries:")
    cluster_counts = Counter(adjusted_labels)
    for cluster_id, count in cluster_counts.items():
        if cluster_id == -1:
            print(f"Cluster {cluster_id} (Noise): {count} instances")
        else:
            print(f"Cluster {cluster_id}: {count} instances")

    # Save the final embeddings and labels for further analysis
    np.save("final_embeddings.npy", embeddings_np)
    np.save("final_adjusted_labels.npy", adjusted_labels)
    print("Final embeddings and labels saved!")

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
    sampled_data = sampled_data.sample(n=300, random_state=55)  # Randomly sample 300 rows
    all_texts = sampled_data['TEXT'].tolist()

    # Run iterative training with visualization
    iterative_training(all_texts, max_iterations=4)


if __name__ == "__main__":
    main()