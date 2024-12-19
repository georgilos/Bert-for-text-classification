import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from collections import Counter
import torch
import torch.nn.functional as F


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
    must_link_dict = {i: set() for i in range(n)}
    for i, j in must_link:
        must_link_dict[i].add(j)
        must_link_dict[j].add(i)

    cannot_link_dict = {i: set() for i in range(n)}
    for i, j in cannot_link:
        cannot_link_dict[i].add(j)
        cannot_link_dict[j].add(i)

    # Visit each point
    visited = np.full(n, False)  # Checks if a point is unvisited and not yet part of a cluster

    def expand_cluster(point_idx):
        # Initialising a double-ended queue to store points for Breadth-First Search
        queue = deque([point_idx])  # Add the index of the initial point to it
        cluster_points = []  # Points that will be part of this cluster
        # (BFS as long as there are points in the deque to be explored)
        while queue:
            current_point = queue.popleft()  # Retrieve and remove the leftmost element from the queue
            if visited[current_point]:  # If a point is visited
                continue  # Skip this point
            is_valid = True
            for other_point in cluster_points:
                if current_point in cannot_link_dict[other_point]:
                    is_valid = False
                    break  # Stop checking if violation is found

            if not is_valid:
                # Handle cannot-link violation:
                # labels[current_point] = -1  # Option 1: Assign to noise
                continue  # Option 2: Skip this point and prevent further expansion
            visited[current_point] = True  # When a point passes the constraint check, mark it as
            cluster_points.append(current_point)  # Add point to the cluster_points list

            # Get neighbors of the current point based on pre-calculated distances in distance_matrix.
            # Points within a distance of 'eps' are considered neighbors.
            # np.where(...) returns indices of neighbors (using [0]).
            neighbors = np.where(distance_matrix[current_point] <= eps)[0]

            # Ensure cannot-link constraints are not violated
            valid_neighbors = []
            for neighbor in neighbors:
                # Check for direct cannot-link violations with all points in the cluster
                if any(neighbor in cannot_link_dict[other_point] for other_point in cluster_points):
                    continue  # Skip this neighbor, as it creates a cannot-link violation
                valid_neighbors.append(neighbor)

            # Add must-link neighbors to the cluster and queue
            for p in must_link_dict[current_point]:
                if p not in cluster_points:
                    queue.append(p)

            # If the current point has enough neighbors, include them in the cluster
            if len(valid_neighbors) >= min_samples:
                for neighbor in valid_neighbors:
                    if neighbor not in cluster_points:
                        queue.append(neighbor)

        # Assign the cluster ID to all points in the cluster
        for p in cluster_points:
            labels[p] = cluster_id

    for i in range(n):
        if visited[i] or labels[i] != -1:  # Skip visited or already clustered points
            continue

        # Check if the point is a core point
        neighbors = np.where(distance_matrix[i] <= eps)[0]
        if len(neighbors) < min_samples:
            continue  # Not a core point, remains noise

        # Expand the cluster
        expand_cluster(i)
        cluster_id += 1

    return labels


def initialize_memory_bank(embeddings, cluster_labels):
    """
    Initialize the memory bank with cluster centroids (PyTorch version).

    Parameters:
    - embeddings (torch.Tensor): Embeddings of all data points (n x d).
    - cluster_labels (torch.Tensor): Cluster labels for each data point.

    Returns:
    - memory_bank (dict): A dictionary where keys are cluster IDs and values are centroids.
    """
    memory_bank = {}  # Initialize memory_bank as a dictionary
    # Ensure cluster_labels are on the same device as embeddings:
    cluster_labels = cluster_labels.to(embeddings.device)
    unique_clusters = torch.unique(cluster_labels[cluster_labels != -1])  # Exclude noise (-1)
    for cluster in unique_clusters:
        cluster_indices = torch.where(cluster_labels == cluster)[0]  # Find points in the cluster
        cluster_embeddings = embeddings[cluster_indices]  # Get embeddings for the cluster
        centroid = cluster_embeddings.mean(dim=0)  # Compute centroid
        centroid = F.normalize(cluster_embeddings.mean(dim=0), p=2, dim=0)
        memory_bank[int(cluster.item())] = centroid  # Store as tensor in memory bank
    return memory_bank


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

    all_texts = sampled_data['TEXT'].tolist()

    # Prompt the user for parameters
    try:
        eps = float(input("\nEnter the value for eps (e.g., 0.13): "))
        min_samples = int(input("Enter the value for min_samples (e.g., 2): "))
    except ValueError:
        print("Invalid input! Please enter a valid float for eps and an integer for min_samples.")
        exit()

    # Initialing empty ML & CL lists
    must_link_pairs = []
    cannot_link_pairs = []

    # Apply constrained DBSCAN
    adjusted_labels = constrained_dbscan_with_constraints(
        distance_matrix, eps, min_samples, must_link_pairs, cannot_link_pairs
    )

    # Calculate and print noise points
    noise_points = [i for i, label in enumerate(adjusted_labels) if label == -1]

    # Reduce distance matrix to 2D for visualization
    reduced_embeddings = PCA(n_components=2).fit_transform(distance_matrix)

    # Scatter plot of clusters
    plt.scatter(
        reduced_embeddings[:, 0], reduced_embeddings[:, 1],
        c=adjusted_labels, cmap='tab10', s=10
    )
    plt.colorbar(label="Cluster Labels")
    plt.title("Cluster Visualization")

    # Combine embeddings, labels, and texts into a DataFrame
    clustered_data = pd.DataFrame({
        'TEXT': all_texts,
        'CLUSTER': adjusted_labels  # From clustering
    })

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

    # Resetting memory_bank
    memory_bank = {}
    # Making sure that adjusted_labels are PyTorch tensors
    adjusted_labels = torch.tensor(adjusted_labels, dtype=torch.int64)
    # Move all embeddings to the desired device (e.g., CPU):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_embeddings = sampled_embeddings.to(device)
    # Creating memory bank
    memory_bank = initialize_memory_bank(all_embeddings, adjusted_labels)
    print("\nMemory Bank Initialized")
    # Save the memory bank
    torch.save(memory_bank, "memory_bank.pt")
    print("Memory bank saved as 'memory_bank.pt'")


if __name__ == "__main__":
    main()
