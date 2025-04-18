import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from collections import Counter
import torch
import torch.nn.functional as F
import re


def constrained_dbscan_with_constraints(distance_matrix, eps, min_samples, must_link, cannot_link):
    """
    Constrained DBSCAN implementation where constraints are applied during clustering.

    Parameters:
    - distance_matrix (ndarray): Pairwise distance matrix.
    - eps (float): Maximum distance to consider two points as neighbors.
    - min_samples (int): Minimum number of neighbors for a point to be a core point.
    - must_link (list of tuples): List of must-link pairs (index1, index2).
    - cannot_link (list of tuples): List of cannot-link pairs (index1, index2).

    Returns:
    - labels (ndarray): Cluster labels for each point. Noise is labeled as -1.
    """
    n = distance_matrix.shape[0]  # Get the total number of data points (# of cols) in the distance matrix.
    labels = np.full(n, -1)  # Initialize all points as noise (-1)
    cluster_id = 0  # Start with cluster ID 0

    # Convert must-link and cannot-link constraints into dictionaries for quick lookup
    must_link_dict = {}
    for i, j in must_link:
        # If i or j already exists in a group, merge the groups
        group_i = must_link_dict.get(i)
        group_j = must_link_dict.get(j)
        if group_i and group_j:
            if group_i != group_j:  # Merge only if they are different groups
                group_i.update(group_j)
                for k in group_j:
                    must_link_dict[k] = group_i
        elif group_i:
            group_i.add(j)
            must_link_dict[j] = group_i
        elif group_j:
            group_j.add(i)
            must_link_dict[i] = group_j
        else:
            # Create a new group if neither i nor j exists in any group
            new_group = {i, j}
            must_link_dict[i] = new_group
            must_link_dict[j] = new_group

    cannot_link_dict = {}
    for i, j in cannot_link:
        cannot_link_dict.setdefault(i, set()).add(j)
        cannot_link_dict.setdefault(j, set()).add(i)

    # Visit each point
    visited = np.full(n, False)  # Assigning all points as not visited

    def expand_cluster(point_idx):
        # Initialising a double-ended queue to store points for Breadth-First Search
        queue = deque([point_idx])  # Add the index of the initial point to it
        cluster_points = set()  # Points that will be part of this cluster
        # (BFS as long as there are points in the deque to be explored)
        while queue:
            current_point = queue.popleft()  # Retrieve and remove the leftmost element from the queue
            if visited[current_point]:  # If a point is visited
                continue  # Skip this point

            # Cannot link check
            if any(current_point in cannot_link_dict.get(p, set()) for p in cluster_points):
                continue  # Skip this point if it violates cannot-link with any point in the cluster
            # Prioritize must link neighbors
            if current_point in must_link_dict:
                group = must_link_dict[current_point]
                for neighbor in group:
                    if neighbor not in cluster_points and not any(
                            neighbor in cannot_link_dict.get(p, set()) for p in cluster_points):
                        cluster_points.add(neighbor)
                        queue.append(neighbor)

            visited[current_point] = True  # When a point passes the constraint check, mark it as
            cluster_points.add(current_point)  # Add point to the cluster_points list

            # Get neighbors of the current point based on pre-calculated distances in distance_matrix.
            # Points within a distance of 'eps' are considered neighbors.
            # np.where(...) returns indices of neighbors (using [0]).
            neighbors = np.where(distance_matrix[current_point] <= eps)[0]
            valid_neighbors = []
            for neighbor in neighbors:
                if neighbor not in cluster_points and not any(
                        neighbor in cannot_link_dict.get(p, set()) for p in cluster_points):
                    valid_neighbors.append(neighbor)

            # Check min_samples condition before adding neighbors
            if len(valid_neighbors) >= min_samples:
                for neighbor in valid_neighbors:
                    queue.append(neighbor)

        if len(cluster_points) >= min_samples:
            for p in cluster_points:
                labels[p] = cluster_id
        else:  # Treat as noise if the cluster is too small
            for p in cluster_points:
                labels[p] = -1

            # Add must-link neighbors to the cluster and queue
            # for p in must_link_dict[current_point]:
            #     if p not in cluster_points:
            #         queue.append(p)

            # If the current point has enough neighbors, include them in the cluster
            # if len(valid_neighbors) >= min_samples:
            #     for neighbor in valid_neighbors:
            #         if neighbor not in cluster_points:
            #             queue.append(neighbor)

        # Assign the cluster ID to all points in the cluster
        # for p in cluster_points:
        #     labels[p] = cluster_id

    # Constraint-based initialization
    # for i, j in must_link:
    #     if labels[i] == -1 and labels[j] == -1:  # Both points are not yet clustered
    #         expand_cluster(i)  # Start a new cluster with the must-link pair
    #         cluster_id += 1
    # Iterate over points
    for i in range(n):
        if visited[i] or labels[i] != -1:  # Skip visited or already clustered points
            continue

        # Check if the point is a core point
        # neighbors = np.where(distance_matrix[i] <= eps)[0]
        # if len(neighbors) < min_samples:
        #     continue  # Not a core point, remains noise

        # Expand the cluster
        expand_cluster(i)
        cluster_id += 1  # Increment cluster ID for the next cluster

    return labels


def compute_cluster_centroids(embeddings, cluster_labels):
    """
    Parameters:
    - embeddings (torch.Tensor): Embeddings of all data points (n x d).
    - cluster_labels (torch.Tensor): Cluster labels for each data point.

    Returns:
    - Cluster_centroids (dict): A dictionary where keys are cluster IDs and values are centroids.
    """
    centroids = {}  # Initialize dict

    # Ensure cluster_labels are on the same device as embeddings:
    cluster_labels = cluster_labels.to(embeddings.device)

    # Exclude noise (-1)
    unique_clusters = torch.unique(cluster_labels[cluster_labels != -1])

    for cluster in unique_clusters:

        # Convert a scalar tensor, into a standard PyTorch tensor
        cluster_tensor = torch.tensor(cluster.item(), dtype=torch.int64, device=cluster_labels.device)

        # Find the indices of the data points that belong to the current cluster
        cluster_indices = torch.where(cluster_labels == cluster_tensor)[0]  # Find points in the cluster
        # cluster_indices = torch.where(cluster_labels == cluster)[0]

        # Get embeddings for the cluster
        cluster_embeddings = embeddings[cluster_indices]

        # Compute centroid
        centroid = cluster_embeddings.mean(dim=0)

        # Normalize centroid
        print("Pre-normalization", torch.norm(centroid, p=2, dim=0))
        centroid = F.normalize(centroid, p=2, dim=0)
        print("Post-normalization", torch.norm(centroid, p=2, dim=0))

        # Store as tensor in memory bank
        centroids[int(cluster.item())] = centroid

    return centroids


def relabel_clusters(labels):
    """Relabels cluster IDs to be consecutive integers starting from 0."""
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]  # Exclude noise

    new_labels = np.full_like(labels, -1)  # Initialize with noise
    for i, cluster_id in enumerate(unique_labels):
        new_labels[labels == cluster_id] = i  # Assign new consecutive IDs
    return new_labels


def main():

    # Load the distance matrix
    try:
        distance_matrix = np.load("distance_matrix.npy")
        print("\nDistance matrix loaded successfully.")
    except FileNotFoundError:
        print("Error: 'distance_matrix.npy' file not found. Make sure the file is in the working directory.")
        exit()

    # Load the sampled_embeddings for the memory bank
    try:
        sampled_embeddings = torch.load("sampled_embeddings.pt")
        print("Sampled embeddings loaded successfully.")
    except FileNotFoundError:
        print("Error: 'sampled_embeddings.pt' file not found. Make sure the file is in the working directory.")
        exit()

    # Load the sampled data
    try:
        sampled_data = pd.read_csv("sampled_data.csv")
        print("Sampled data loaded successfully.")
    except FileNotFoundError:
        print("Error: 'sampled_data.csv' file not found. Make sure the file is in the working directory.")
        exit()

    # Prompt the user for parameters
    try:
        eps = float(input("\nEnter the value for eps (e.g., 0.13): "))
        min_samples = int(input("Enter the value for min_samples (e.g., 2): "))
    except ValueError:
        print("Invalid input! Please enter a valid float for eps and an integer for min_samples.")
        exit()

    # Initialing ML & CL lists. Add the ML (0,9) pair for a conflict
    # must_link_pairs = [(0,1),(1,2),(2,3),(3,5),(4,6),(6,9)]
    # cannot_link_pairs = [(5,4)]  # np.load("cannot_link_pairs.npy", allow_pickle=True).tolist()

    must_link_pairs = []
    cannot_link_pairs = []

    # Apply constrained DBSCAN
    adjusted_labels = constrained_dbscan_with_constraints(
        distance_matrix, eps, min_samples, must_link_pairs, cannot_link_pairs
    )

    # Relabeling clusters
    adjusted_labels = relabel_clusters(adjusted_labels)

    # Calculate and print noise points
    noise_points = [i for i, label in enumerate(adjusted_labels) if label == -1]

    # Reduce distance matrix to 2D for visualization
    from sklearn.decomposition import PCA
    reduced_embeddings = PCA(n_components=2).fit_transform(distance_matrix)

    # Reduce distance matrix to 2D for visualization using t-SNE
    # from sklearn.manifold import TSNE
    # reduced_embeddings = TSNE(n_components=2, random_state=42).fit_transform(distance_matrix)

    # Reduce distance matrix to 2D for visualization using UMAP
    # import umap
    # reducer = umap.UMAP(n_components=2, random_state=42)
    # reduced_embeddings = reducer.fit_transform(distance_matrix)

    # Scatter plot of clusters
    plt.scatter(
        reduced_embeddings[:, 0], reduced_embeddings[:, 1],
        c=adjusted_labels, cmap='tab10', s=10
    )
    plt.colorbar(label="Cluster Labels")
    plt.title("Cluster Visualization")

    # Create incremental IDs
    incremental_ids = list(range(len(sampled_data)))

    # Add IDs as text annotations next to each point
    for i, point in enumerate(reduced_embeddings):
        plt.annotate(str(incremental_ids[i]),  # Use ID from sampled_data
                     xy=(point[0], point[1]),
                     xytext=(5, 5),  # Offset the label slightly
                     textcoords='offset points',
                     fontsize=8)

    # Combine embeddings, labels, and texts into a DataFrame
    all_texts = sampled_data['TEXT'].tolist()
    clustered_data = pd.DataFrame({
        'TEXT': all_texts,
        'CLUSTER': adjusted_labels  # From clustering
    })

    # The following line is added to resolve some unreadable character encoding problems
    clustered_data['TEXT'] = clustered_data['TEXT'].apply(lambda text: re.sub(r'\ufe0f', '', text))

    # Group by clusters and inspect
    for cluster_id in np.unique(adjusted_labels):
        pd.set_option('display.max_colwidth', None)  # Set to None for unlimited width
        print(f"\nCluster {cluster_id}:")
        print(clustered_data[clustered_data['CLUSTER'] == cluster_id].head(15))  # Inspect first 15 texts

    # Count the instances in each cluster
    cluster_counts = Counter(adjusted_labels)

    # Save the adjusted labels
    np.save("adjusted_labels.npy", adjusted_labels)
    print("\nClustering complete. Labels saved as 'adjusted_labels.npy'")

    # Save the figure
    output_file = "cluster_visualization.png"
    plt.savefig(output_file)
    print(f"Cluster visualization saved as '{output_file}'")

    # Print Results
    print("\nAdjusted Cluster Labels:", np.unique(adjusted_labels))
    print(f"Number of Noise Points: {len(noise_points)}")
    # Print the cluster counts
    for cluster_id, count in cluster_counts.items():
        print(f"Cluster {cluster_id}: {count} instances")

    # Making sure that adjusted_labels are PyTorch tensors
    adjusted_labels = torch.tensor(adjusted_labels, dtype=torch.int64)
    # Move all embeddings to the desired device (e.g., CPU):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_embeddings = sampled_embeddings.to(device)
    # Creating memory bank
    centroids = compute_cluster_centroids(all_embeddings, adjusted_labels)
    # Save centroids
    torch.save(centroids, "centroids.pt")
    print("\nCentroids gathered and saved as 'centroids.pt' ")


if __name__ == "__main__":
    main()
