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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.train()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print("Generating initial embeddings...")
    sampled_embeddings = generate_embeddings(all_texts, tokenizer, model, batch_size=16, use_cls=True)
    embeddings_np = sampled_embeddings.cpu().numpy()
    labels_placeholder = np.zeros(len(all_texts))
    plot_embeddings(embeddings_np, labels_placeholder, iteration=0, stage='before', title="Initial Embeddings")
    distance_matrix = cdist(embeddings_np, embeddings_np, metric='cosine')
    mean_distance = np.mean(distance_matrix)
    print(f"Mean Pairwise Distance: {mean_distance:.4f}")

    while True:
        eps = float(input(
            f"Enter the eps value for initial clustering (default suggestion: {mean_distance * 0.5:.4f}): ") or mean_distance * 0.5)
        min_samples = int(input(f"Enter the min_samples value for initial clustering (default suggestion: 2): ") or 2)
        adjusted_labels, cannot_link_dict = constrained_dbscan_with_constraints(
            distance_matrix, eps, min_samples, must_link=[], cannot_link=[]
        )
        adjusted_labels = merge_small_clusters(distance_matrix, adjusted_labels, cannot_link_dict, min_samples)
        unique_clusters = np.unique(adjusted_labels)
        print(f"\nClusters Found: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)}")
        cluster_counts = Counter(adjusted_labels)
        for cluster_id, count in cluster_counts.items():
            print(f"Cluster {cluster_id}: {count} instances")
        user_choice = input("Are you satisfied with the initial clustering? (y/n): ").strip().lower()
        if user_choice == "y":
            break

    plot_embeddings(embeddings_np, adjusted_labels, iteration=0, stage='after', title="Embeddings After Initial Clustering")
    memory_bank = initialize_memory_bank(sampled_embeddings, torch.tensor(adjusted_labels, dtype=torch.int64))
    must_link_pairs, cannot_link_pairs = [], []
    must_link_inclusion = {}
    cannot_link_inclusion = {}

    for iteration in range(1, max_iterations + 1):
        print(f"\nIteration {iteration}:")
        uncertain_positive_pairs, uncertain_negative_pairs = select_uncertain_pairs(distance_matrix, adjusted_labels)
        must_link_pairs, cannot_link_pairs = annotate_and_update_constraints(
            uncertain_positive_pairs, uncertain_negative_pairs, all_texts, must_link_pairs, cannot_link_pairs
        )

        # Add newly annotated pairs to the inclusion dictionaries
        for pair in must_link_pairs:
            if pair not in must_link_inclusion:
                must_link_inclusion[pair] = 0
        for pair in cannot_link_pairs:
            if pair not in cannot_link_inclusion:
                cannot_link_inclusion[pair] = 0

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        model.train()

        for epoch in range(1):
            for i in range(0, len(all_texts), 32):
                batch_indices = list(range(i, min(i + 32, len(all_texts))))
                prioritized_must_link = sorted(must_link_pairs, key=lambda pair: must_link_inclusion[pair])
                prioritized_cannot_link = sorted(cannot_link_pairs, key=lambda pair: cannot_link_inclusion[pair])
                prioritized_indices = set()
                for a, b in prioritized_must_link[:16]:
                    prioritized_indices.update([a, b])
                for a, b in prioritized_cannot_link[:16]:
                    prioritized_indices.update([a, b])
                batch_indices = list(prioritized_indices.union(batch_indices))
                batch_indices = batch_indices[:32]
                batch_index_set = set(batch_indices)
                batch_texts = [all_texts[idx] for idx in batch_indices]
                inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
                inputs = {key: val.to(device) for key, val in inputs.items()}
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]

                index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(batch_indices)}
                batch_must_link_pairs = [
                    (index_map[a], index_map[b]) for (a, b) in must_link_pairs if a in batch_index_set and b in batch_index_set
                ]
                batch_cannot_link_pairs = [
                    (index_map[a], index_map[b]) for (a, b) in cannot_link_pairs if a in batch_index_set and b in batch_index_set
                ]

                contrastive_loss = calculate_contrastive_loss(memory_bank, embeddings,
                                                              torch.tensor(adjusted_labels)[batch_indices], temperature)
                pairwise_distances = torch.cdist(embeddings, embeddings, p=2)

                support_pair_loss = calculate_support_pair_loss(embeddings, batch_must_link_pairs,
                                                                batch_cannot_link_pairs, margin)
                combined_loss = contrastive_loss + lambda_t * support_pair_loss
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()

                # Ensure all pairs in the batch are updated in the inclusion dictionaries
                for a, b in batch_must_link_pairs:
                    must_link_inclusion[(a, b)] += 1
                for a, b in batch_cannot_link_pairs:
                    cannot_link_inclusion[(a, b)] += 1

        print("Recomputing embeddings and clustering...")
        updated_embeddings = generate_embeddings(all_texts, tokenizer, model, batch_size=16, use_cls=True)
        embeddings_np = updated_embeddings.cpu().numpy()
        distance_matrix = cdist(embeddings_np, embeddings_np, metric='cosine')
        eps = float(input(f"Enter the eps value for clustering (default suggestion: {mean_distance * 0.5:.4f}): ") or mean_distance * 0.5)
        min_samples = int(input(f"Enter the min_samples value for clustering (default suggestion: 2): ") or 2)
        adjusted_labels, cannot_link_dict = constrained_dbscan_with_constraints(
            distance_matrix, eps, min_samples, must_link_pairs, cannot_link_pairs
        )
        adjusted_labels = merge_small_clusters(distance_matrix, adjusted_labels, cannot_link_dict, min_samples)
        plot_embeddings(embeddings_np, adjusted_labels, iteration=iteration, stage='after', title="Embeddings After Clustering")
        memory_bank = initialize_memory_bank(updated_embeddings, torch.tensor(adjusted_labels, dtype=torch.int64))

    np.save("final_embeddings.npy", embeddings_np)
    np.save("final_adjusted_labels.npy", adjusted_labels)
    save_path = "./models/fine_tuned_bert.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuned model saved to {save_path}")

def main():
    data_path = "data/unlabeled_data/cleaned_texts_unlabeled.csv"
    sampled_data = pd.read_csv(data_path, header=None)
    sampled_data.columns = ['ID', 'TEXT']
    sampled_data['TEXT'] = sampled_data['TEXT'].fillna('').astype(str).str.replace(r'[\ufe0f\x0f]', '', regex=True)
    sampled_data = sampled_data[sampled_data['TEXT'].str.strip() != '']
    sampled_data = sampled_data.sample(n=300, random_state=55)
    all_texts = sampled_data['TEXT'].tolist()
    iterative_training(all_texts, max_iterations=4)

if __name__ == "__main__":
    main()
