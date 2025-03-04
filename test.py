import os
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from collections import Counter
from scipy.spatial.distance import cdist
from constrained_clustering import constrained_dbscan_with_constraints, relabel_clusters, compute_cluster_centroids
from uncertain_pairs import select_uncertain_pairs, annotate_and_update_constraints
from data_embeddings_distance_mat import generate_embeddings
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
# import os
import random
import json


def save_k_distance_plot(embeddings, k=5, save_path="images/elbow_plot.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
    neighbors = NearestNeighbors(n_neighbors=k, metric='cosine')
    neighbors_fit = neighbors.fit(embeddings)
    distances, _ = neighbors_fit.kneighbors(embeddings)
    distances = np.sort(distances[:, k - 1], axis=0)

    plt.figure(figsize=(8, 6))
    plt.plot(distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}th Nearest Neighbor Distance')
    plt.title('k-Distance Graph')
    plt.savefig(save_path)
    print(f"Elbow plot saved at: {save_path}")
    plt.close()  # Free resources
    return distances


def visualize_clusters(embeddings, cluster_labels, identifier, method='pca', iteration=None):
    """
    Visualize clusters using PCA, t-SNE, or UMAP and save the plot to the "images" folder.

    Parameters:
    - embeddings: Tensor containing the embeddings of all instances.
    - cluster_labels: Cluster labels for each instance (e.g., output of constrained DBSCAN).
    - identifier: List or array of instance IDs (e.g., numerical IDs or indices).
    - method: Dimensionality reduction method ('pca', 'tsne', or 'umap').
    - iteration: Current iteration number for naming the saved image file.
    """
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid method. Choose 'pca', 'tsne', or 'umap'.")

    # Reduce embeddings to 2D
    reduced_embeddings = reducer.fit_transform(embeddings.cpu().numpy())

    # Create a color map for the clusters
    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels)
    colors = plt.colormaps['tab10'].resampled(num_clusters)  # Use a colormap with enough colors

    # Plot the clusters
    plt.figure(figsize=(12, 10))
    for i, label in enumerate(unique_labels):
        # Get the points belonging to the current cluster
        cluster_mask = (cluster_labels == label)
        cluster_points = reduced_embeddings[cluster_mask]
        cluster_identifiers = np.array(identifier)[cluster_mask]  # Get IDs for points in this cluster

        # Plot the points with the corresponding color
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=colors(i), label=f'Cluster {label}', s=10)

        # Annotate each point with its instance ID
        for idx, (x, y) in enumerate(cluster_points):
            plt.text(x, y, str(cluster_identifiers[idx]), fontsize=8, ha='right', va='bottom')

    plt.title(f"Cluster Visualization ({method.upper()}) - Iteration {iteration}")
    plt.legend()  # Add a legend to show cluster labels

    # Save the plot to the "images" folder
    os.makedirs("images", exist_ok=True)
    save_path = f"images/cluster_visualization_iteration_{iteration}.png"
    plt.savefig(save_path)
    print(f"Cluster visualization saved at: {save_path}")
    plt.close()  # Free resources


def calculate_contrastive_loss(centroids, embeddings, cluster_labels, temperature=0.05):
    """
    Calculate the contrastive loss (L_c) based on instance-to-centroid contrastive loss.
    """
    # Filter out noise points (cluster label == -1)
    valid_indices = cluster_labels != -1
    valid_embeddings = embeddings[valid_indices]

    valid_labels = cluster_labels[valid_indices]
    valid_embeddings = F.normalize(valid_embeddings, p=2, dim=1)
    """""

    # Prepare the centroids for all valid points
    centroids = torch.stack([centroids[label.item()] for label in valid_labels])
    # centroids = F.normalize(selected_centroids, p=2, dim=0)

    # Calculate logits: instance-to-centroid similarity
    logits = torch.mm(valid_embeddings, centroids.T) / temperature

    # Create the labels for the contrastive loss
    labels = valid_labels.to(logits.device)
    """

    # Correctly stack centroids based on unique labels
    # centroids_tensor = torch.stack([centroids[label.item()] for label in unique_labels])
    # centroids_tensor = F.normalize(centroids_tensor, p=2, dim=1)  # Normalize
    # Assuming 'centroids' is your dictionary
    all_centroids = torch.stack(list(centroids.values()))

    # Normalize
    centroids_tensor = F.normalize(all_centroids, p=2, dim=1)  # Normalize

    # Calculate logits (scaled cosine similarity between embeddings and the cluster centroids)
    logits = torch.mm(valid_embeddings, centroids_tensor.T) / temperature
    """""
    # Create labels for CrossEntropyLoss
    labels = torch.zeros_like(valid_labels)
    # Get unique cluster labels
    unique_labels = torch.unique(valid_labels)
    for i, label in enumerate(unique_labels):
        labels[valid_labels == label] = i
    """""
    # Move labels to the same device as logits
    valid_labels = valid_labels.to(logits.device)
    valid_labels = valid_labels.to(torch.int64)
    # Calculate contrastive loss using CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, valid_labels)

    return loss


def calculate_support_pair_loss(embeddings, must_link_pairs, cannot_link_pairs, distance_matrix, batch_indices, margin=1.0, debug=False):
    """
    Calculate the Support Pair Constraints Loss (L_t) based on must-link and cannot-link constraints.
    """
    # Normalize embeddings (einai hdh normalized, mallon prepei na kanw remove)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Precompute pairwise distances
    # pairwise_distances = torch.tensor(distance_matrix, device=embeddings.device)

    triplet_losses = []

    # Identifying Positive and Negative Indexes
    for anchor_idx in range(embeddings.size(0)):
        positives = [j for (a, j) in must_link_pairs if a == anchor_idx] + \
                    [a for (a, j) in must_link_pairs if j == anchor_idx]
        negatives = [j for (a, j) in cannot_link_pairs if a == anchor_idx] + \
                    [a for (a, j) in cannot_link_pairs if j == anchor_idx]

        if not positives or not negatives:
            continue
        """""
        # Find the hardest positive and hardest negative
        hardest_positive_idx = torch.argmax(torch.tensor([pairwise_distances[anchor_idx, p] for p in positives]))
        hardest_positive = positives[hardest_positive_idx]
        """""

        # Compute distances for must-link pairs
        # positive_distances = [pairwise_distances[anchor_idx, p] for p in positives]
        # Use global distance matrix for must-link pairs
        positive_distances = [distance_matrix[batch_indices[anchor_idx], batch_indices[p]] for p in positives]

        # Debug: Print all positive distances
        print(f"Anchor is located in the index: {anchor_idx}, Must-Link Distances: {positive_distances}")

        # Select the hardest positive (largest distance)
        hardest_positive_idx = torch.argmax(torch.tensor(positive_distances, dtype=torch.float32))
        hardest_positive = positives[hardest_positive_idx]

        # Debug: Print the selected hardest positive and its distance
        print(
            f"Selected Hardest Positive Pair with local indexes: ({anchor_idx}, {hardest_positive}), Distance: {positive_distances[hardest_positive_idx]:.4f}")

        # Compute global distances for cannot-link pairs
        # anchor_global_idx = batch_indices[anchor_idx]
        negative_distances = [distance_matrix[batch_indices[anchor_idx], batch_indices[n]] for n in negatives]

        # Debug: Print all negative distances
        print(f"Anchor is located in the index: {anchor_idx}, Cannot-Link Distances: {negative_distances}")

        # Select the hardest negative (smallest global distance)
        hardest_negative_idx = torch.argmin(torch.tensor(negative_distances, dtype=torch.float32))
        hardest_negative = negatives[hardest_negative_idx]

        # Debug: Print the selected hardest negative and its distance
        print(
            f"Selected Hardest Negative Pair: ({anchor_idx}, {hardest_negative}), Distance: {negative_distances[hardest_negative_idx]:.4f}")
        hardest_positive_global_idx = batch_indices[hardest_positive]
        hardest_negative_global_idx = batch_indices[hardest_negative]

        # Compute triplet loss
        # Replace batch-local distance computation with global distance lookup
        positive_distance = torch.tensor(distance_matrix[batch_indices[anchor_idx], hardest_positive_global_idx],dtype=torch.float32,
                                         device=embeddings.device, requires_grad=True)
        negative_distance = torch.tensor(distance_matrix[batch_indices[anchor_idx], hardest_negative_global_idx],dtype=torch.float32,
                                         device=embeddings.device, requires_grad=True)

        triplet_loss = F.relu(positive_distance - negative_distance + margin)
        triplet_losses.append(triplet_loss)

        # Debug print
        if debug:
            print(f"The anchor of the current batch is in index {anchor_idx}, "
                  f"Hardest Positive Pair: ({anchor_idx}, {hardest_positive}), "
                  f"Positive Distance: {positive_distance.item():.4f}, "
                  f"Hardest Negative Pair: ({anchor_idx}, {hardest_negative}), "
                  f"Negative Distance: {negative_distance.item():.4f}"
                  f" Triplet Loss: {triplet_loss.item():.4f}")

    # Average triplet losses
    if triplet_losses:
        support_pair_loss = torch.stack(triplet_losses).mean()
    else:
        support_pair_loss = torch.tensor(0.0, requires_grad=True)

    return support_pair_loss


def assign_anchors_to_batches(all_texts, anchors, batch_size):
    """
    Create batches prioritizing anchors and their related instances.
    Each anchor and its related instances are assigned to their own batch.
    Remaining instances are shuffled and assigned to batches of fixed size.

    Parameters:
    - all_texts: List of all instances.
    - anchors: Dictionary of anchors and their related instances.
    - batch_size: Size of each batch.

    Returns:
    - batches: List of batches, where each batch is a list of indices.
    """
    all_indices = set(range(len(all_texts)))  # All instance indices
    batches = []

    # Step 1: Assign anchors and their related instances to batches
    for anchor, related_instances in anchors.items():
        # Create a batch with the anchor and its related instances
        batch = [anchor] + list(related_instances)
        batches.append(batch)

    # Step 2: Collect all instances used in anchor-related batches
    used_instances = set()
    for batch in batches:
        used_instances.update(batch)

    # Step 3: Assign remaining instances to batches of fixed size
    remaining_instances = list(all_indices - used_instances)
    random.shuffle(remaining_instances)  # Shuffle remaining instances

    for i in range(0, len(remaining_instances), batch_size):
        batch = remaining_instances[i:i + batch_size]
        batches.append(batch)

    return batches


def debug_pair_distances(embeddings, must_link_pairs, cannot_link_pairs):
    """
    Print the distances of all must-link and cannot-link pairs.

    Parameters:
    - embeddings: Tensor containing the embeddings of all instances.
    - must_link_pairs: List of must-link pairs (tuples of indices).
    - cannot_link_pairs: List of cannot-link pairs (tuples of indices).
    """
    print("\nDistances of Must-Link and Cannot-Link Pairs:")
    # Normalize embeddings for cosine similarity
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    for a, b in must_link_pairs:
        distance = 1 - torch.dot(normalized_embeddings[a], normalized_embeddings[b]).item()
        print(f"Distance of ML({a},{b}): {distance:.4f}")
    for a, b in cannot_link_pairs:
        distance = 1 - torch.dot(normalized_embeddings[a], normalized_embeddings[b]).item()
        print(f"Distance of CL({a},{b}): {distance:.4f}")


def find_anchors(must_link_pairs, cannot_link_pairs):
    """
    Identify anchors and their related instances.

    Parameters:
    - must_link_pairs: List of tuples representing must-link pairs.
    - cannot_link_pairs: List of tuples representing cannot-link pairs.

    Returns:
    - Dictionary of anchors with related instances.
    """
    anchors = {}

    # Collect relationships
    for pair in must_link_pairs:
        for instance in pair:
            if instance not in anchors:
                anchors[instance] = {'must_link': set(), 'cannot_link': set()}
            anchors[instance]['must_link'].update(pair)

    for pair in cannot_link_pairs:
        for instance in pair:
            if instance not in anchors:
                anchors[instance] = {'must_link': set(), 'cannot_link': set()}
            anchors[instance]['cannot_link'].update(pair)

    # Filter true anchors
    true_anchors = {}
    for anchor, relations in anchors.items():
        if relations['must_link'] and relations['cannot_link']:
            related_instances = relations['must_link'].union(relations['cannot_link'])
            related_instances.discard(anchor)
            true_anchors[anchor] = related_instances

    return true_anchors


def iterative_training(all_texts, max_iterations=50, margin=1.0, temperature=1.0, lambda_t=1.0, batch_size=32):
    """
    Perform iterative training with dynamic eps and min_samples selection.
    """
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Step 1: Generate Initial Embeddings
    print("Generating initial embeddings...")
    sampled_embeddings = generate_embeddings(all_texts, tokenizer, model, batch_size=32)
    distance_matrix = cdist(sampled_embeddings.cpu().numpy(), sampled_embeddings.cpu().numpy(), metric='cosine')
    # Set diagonal to 0
    np.fill_diagonal(distance_matrix, 0)

    # Save the elbow plot for initial embeddings
    # save_k_distance_plot(sampled_embeddings.cpu().numpy(), k=5, save_path="images/initial_elbow_plot.png")

    # Calculate and display mean pairwise distance
    mean_distance = np.mean(distance_matrix)
    print(f"Mean Pairwise Distance: {mean_distance:.4f}")

    # Dynamic parameter adjustment for initial clustering
    while True:
        eps = float(input(
            f"Enter the eps value for initial clustering (default suggestion: {mean_distance * 0.5:.4f}): ") or mean_distance * 0.5)
        min_samples = int(input(f"Enter the min_samples value for initial clustering (default suggestion: 2): ") or 2)

        # Initialing empty ML & CL lists
        must_link_pairs = []  # np.load("must_link_pairs.npy",allow_pickle=True).tolist()
        cannot_link_pairs = []  # np.load("cannot_link_pairs.npy", allow_pickle=True).tolist()

        # Initialing empty ML & CL lists
        # must_link_pairs = [(0, 1), (1, 2), (2, 3), (3, 5), (4, 6), (6, 9)]
                             # np.load("must_link_pairs.npy",allow_pickle=True).tolist()
        # cannot_link_pairs = [(5, 4)]

        # Perform initial clustering
        adjusted_labels = constrained_dbscan_with_constraints(distance_matrix, eps, min_samples, must_link_pairs,
                                                              cannot_link_pairs)
        # Relabeling clusters
        adjusted_labels = relabel_clusters(adjusted_labels)
        # print("Now merging clusters with instance<min_samples")
        # adjusted_labels = merge_small_clusters(distance_matrix, adjusted_labels, cannot_link_dict, min_samples)
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
            print("Re clustering with different parameters...")

    # Compute centroids
    centroids = compute_cluster_centroids(sampled_embeddings, torch.tensor(adjusted_labels, dtype=torch.int64))

    # edw evaza pairs gia dhmiourgia merikwn anchors

    # Iterative process
    model.train()  # Set model to training mode
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration}:")

        # Step 2: Select Uncertain Pairs and Annotate
        uncertain_positive_pairs, uncertain_negative_pairs = select_uncertain_pairs(distance_matrix, adjusted_labels, must_link_pairs, cannot_link_pairs)
        must_link_pairs, cannot_link_pairs = annotate_and_update_constraints(
            uncertain_positive_pairs, uncertain_negative_pairs, all_texts, must_link_pairs, cannot_link_pairs
        )

        # Call the function to find anchors
        anchors = find_anchors(must_link_pairs, cannot_link_pairs)

        # Print results
        if anchors:
            print("Anchors found:")
            for anchor, related_instances in anchors.items():
                print(f"Anchor: {anchor}, Related Instances: {list(related_instances)}")
        else:
            print("No anchors found.")

        # Call the debug_pair_distances function that prints the distances
        debug_pair_distances(sampled_embeddings, must_link_pairs, cannot_link_pairs)

        # Step 3: Fine-Tune Model
        print("Fine-tuning model...")

        batches = assign_anchors_to_batches(all_texts, anchors, batch_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=35e-5, weight_decay=1e-5)

        # Adding epochs
        num_epochs = 1  # Define the number of epochs per iteration
        alpha = 0.25  # Momentum coefficient for memory bank updates
        for epoch in range(num_epochs):  # Start epoch loop
            print(f"Epoch {epoch + 1}/{num_epochs}")
            for batch_indices in batches:  # Use the precomputed batches
                batch_texts = [all_texts[idx] for idx in batch_indices]
                batch_index_set = set(batch_indices)

                # Tokenize and generate embeddings
                # inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
                # inputs = {key: val.to(device) for key, val in inputs.items()}
                # outputs = model(**inputs)
                # batch_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token embeddings

                # TRY UNCOMMENTING THE FOLLOWING 2 LINES AND COMMENTING THE NEXT BATCH EMBEDDINGS

                batch_embeddings = sampled_embeddings[batch_indices]
                batch_embeddings.requires_grad = True

                # Enable gradients locally for batch embeddings (THIS NEXT ONE TO COMMENT)
                # batch_embeddings = sampled_embeddings[batch_indices].clone().detach().requires_grad_(True)

                # Filter must-link and cannot-link pairs whose instances are in the batch
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

                # Log global distances for all batch pairs
                print("Global Distances for Batch:")
                for a, b in batch_must_link_pairs + batch_cannot_link_pairs:
                    global_a, global_b = batch_indices[a], batch_indices[b]
                    print(f"Global Pair ({global_a}, {global_b}), Distance: {distance_matrix[global_a, global_b]:.4f}")

                # Compute hybrid loss
                contrastive_loss = calculate_contrastive_loss(centroids, batch_embeddings,
                                                              torch.tensor(adjusted_labels)[batch_indices],
                                                              temperature)

                support_pair_loss = calculate_support_pair_loss(batch_embeddings, batch_must_link_pairs,
                                                                batch_cannot_link_pairs,
                                                                distance_matrix,
                                                                batch_indices,
                                                                margin,
                                                                debug=False)
                combined_loss = contrastive_loss + lambda_t * support_pair_loss

                # Log the losses for monitoring
                print(f"Iteration {iteration}, Epoch {epoch + 1}, Batch {batches.index(batch_indices) + 1}: "
                      f"Contrastive Loss = {contrastive_loss.item():.4f}, "
                      f"Support Pair Loss = {support_pair_loss.item():.4f}, "
                      f"Combined Loss = {combined_loss.item():.4f}")
                print(f"Batch {batches.index(batch_indices) + 1}: {batch_indices}")
                print(f"Must-Link Pairs with Batch Local indexes: {batch_must_link_pairs}")
                print(f"Cannot-Link Pairs with Batch Local indexes: {batch_cannot_link_pairs}")

                # Update model
                optimizer.zero_grad()

                # print("Gradients before backward pass:")
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(name, param.grad)

                combined_loss.backward()

                # print("Gradients after backward pass:")
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(name, param.grad)

                # print("Model parameters before update:")
                # for name, param in model.named_parameters():
                #     print(name, param)

                optimizer.step()

                # print("Model parameters after update:")
                # for name, param in model.named_parameters():
                #     print(name, param)

                # print("Embeddings before update:")
                # print(sampled_embeddings)

                # Recompute embeddings for the batch using the updated model
                with torch.no_grad():
                    updated_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt",
                                               max_length=128)
                    updated_inputs = {key: val.to(device) for key, val in updated_inputs.items()}
                    updated_outputs = model(**updated_inputs)
                    updated_batch_embeddings = updated_outputs.pooler_output # updated_outputs.last_hidden_state[:, 0, :]

                    print("Pre-normalization", torch.norm(updated_batch_embeddings, p=2, dim=1))
                    updated_batch_embeddings = F.normalize(updated_batch_embeddings, p=2, dim=1)  # L2 normalization
                    print("Post-normalization", torch.norm(updated_batch_embeddings, p=2, dim=1))

                # Update memory bank using momentum
                for local_idx, global_idx in enumerate(batch_indices):
                    sampled_embeddings[global_idx] = (
                            alpha * sampled_embeddings[global_idx] + (1 - alpha) * updated_batch_embeddings[local_idx]
                    )

                print("Batch Embeddings updated")
                # print(sampled_embeddings)
                print("------------NEXT BATCH------------")

        # Visualize clusters after each iteration
        visualize_clusters(sampled_embeddings, adjusted_labels, range(len(sampled_embeddings)), method='pca', iteration=iteration)

        # Step 4: Recompute Embeddings, Clusters, and Memory Bank
        print("Recomputing embeddings and clustering...")
        # updated_embeddings = generate_embeddings(all_texts, tokenizer, model, batch_size=16, use_cls=True)
        distance_matrix = cdist(sampled_embeddings.cpu().numpy(), sampled_embeddings.cpu().numpy(), metric='cosine')
        # Set diagonal to 0
        np.fill_diagonal(distance_matrix, 0)

        # Save the elbow plot for updated embeddings
        # save_k_distance_plot(sampled_embeddings.cpu().numpy(), k=5,
        #                      save_path=f"images/elbow_plot_iteration_{iteration + 1}.png")

        # Calculate and display mean pairwise distance
        mean_distance = np.mean(distance_matrix)
        print(f"Mean Pairwise Distance (Iteration {iteration + 1}): {mean_distance:.4f}")

        # Dynamic parameter adjustment for clustering
        while True:
            eps = float(input(
                f"Enter the eps value for clustering (default suggestion: {mean_distance * 0.5:.4f}): ") or mean_distance * 0.5)
            min_samples = int(input(f"Enter the min_samples value for clustering (default suggestion: 2): ") or 2)

            # Perform clustering
            adjusted_labels = constrained_dbscan_with_constraints(distance_matrix, eps, min_samples, must_link_pairs,
                                                                  cannot_link_pairs)
            # Relabeling clusters
            adjusted_labels = relabel_clusters(adjusted_labels)
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
                print("Re clustering with different parameters...")

        centroids = compute_cluster_centroids(sampled_embeddings, torch.tensor(adjusted_labels, dtype=torch.int64))

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
    torch.save(centroids, "./models/final_centroids.pt")
    # Convert numpy.int64 keys to int
    cluster_labels = {int(k): v for k, v in cluster_labels.items()}
    with open("./models/cluster_labels.json", "w") as f:
        json.dump(cluster_labels, f)

    print("Cluster labels saved!")
    print(cluster_labels)
    # Save the fine-tuned model
    save_path = "./models/fine_tuned_bert.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuned model saved to {save_path}")


def main():
    data_path = "data/unlabeled_data/unlabeled_kaggle_texts.csv"  # data/unlabeled_data/unlabeled_cnn_texts.csv
    sampled_data = pd.read_csv(data_path, header=None)  # Load the CSV file (no headers)
    sampled_data.columns = ['ID', 'TEXT']  # Add column names to the CSV

    # Clean and validate TEXT column
    sampled_data['TEXT'] = sampled_data['TEXT'].fillna('').astype(str).str.replace(r'[\ufe0f\x0f\u0964]', '', regex=True)
    sampled_data = sampled_data[sampled_data['TEXT'].str.strip() != '']

    # Sample and prepare the data
    sampled_data = sampled_data.sample(n=20, random_state=76)  # Randomly sample # rows
    all_texts = sampled_data['TEXT'].tolist()

    # Run iterative training
    iterative_training(all_texts, max_iterations=50, batch_size=32)  # Run algorithm for # repetitions


if __name__ == "__main__":
    main()
