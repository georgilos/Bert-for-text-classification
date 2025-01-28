import numpy as np
import pandas as pd


def select_uncertain_pairs(distance_matrix, cluster_labels, must_link_pairs, cannot_link_pairs):
    """
    Select uncertain positive and negative pairs based on the clustering results and distance matrix.

    Parameters:
    - distance_matrix (ndarray): Pairwise distance matrix (n x n).
    - cluster_labels (ndarray): Cluster labels for each point.

    Returns:
    - uncertain_positive_pairs (list): List of (index1, index2, distance) for uncertain positive pairs.
    - uncertain_negative_pairs (list): List of (index1, index2, distance) for uncertain negative pairs.
    """
    uncertain_positive_pairs = []  # For storing uncertain positive pairs
    uncertain_negative_pairs = []  # For storing uncertain negative pairs

    # Convert constraint lists to sets for faster lookup
    must_link_set = set(must_link_pairs)
    cannot_link_set = set(cannot_link_pairs)

    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])  # Ignore noise (-1)

    # Uncertain Positive Pairs
    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) > 1:
            # Find the pair with the maximum distance within the cluster
            max_dist = -np.inf
            best_pair = None
            for i in cluster_indices:
                for j in cluster_indices:
                    if i < j and (i, j) not in must_link_set and (j, i) not in must_link_set \
                            and (i, j) not in cannot_link_set and (j, i) not in cannot_link_set:
                        dist = distance_matrix[i, j]
                        if dist > max_dist:
                            max_dist = dist
                            best_pair = (i, j, dist)
            if best_pair:
                uncertain_positive_pairs.append(best_pair)

    # Uncertain Negative Pairs
    for cluster1 in unique_clusters:
        for cluster2 in unique_clusters:
            if cluster1 < cluster2:  # Avoid duplicate comparisons
                cluster1_indices = np.where(cluster_labels == cluster1)[0]
                cluster2_indices = np.where(cluster_labels == cluster2)[0]
                # Find the pair with the minimum distance between the two clusters
                min_dist = np.inf
                best_pair = None
                for i in cluster1_indices:
                    for j in cluster2_indices:
                        if (i, j) not in must_link_set and (j, i) not in must_link_set \
                                and (i, j) not in cannot_link_set and (j, i) not in cannot_link_set:
                            dist = distance_matrix[i, j]
                            if dist < min_dist:
                                min_dist = dist
                                best_pair = (i, j, dist)
                if best_pair:
                    uncertain_negative_pairs.append(best_pair)

    return uncertain_positive_pairs, uncertain_negative_pairs


def annotate_and_update_constraints(
    uncertain_positive_pairs, uncertain_negative_pairs,
    all_texts, must_link_pairs, cannot_link_pairs
):
    """
    Annotates uncertain pairs and updates must-link and cannot-link constraints.

    Parameters:
    - uncertain_positive_pairs (list): List of uncertain positive pairs (index1, index2, distance).
    - uncertain_negative_pairs (list): List of uncertain negative pairs (index1, index2, distance).
    - all_texts (list): List of texts corresponding to embeddings.
    - must_link_pairs (list): Existing list of must-link pairs (to be updated).
    - cannot_link_pairs (list): Existing list of cannot-link pairs (to be updated).

    Returns:
    - must_link_pairs: Updated must-link pairs.
    - cannot_link_pairs: Updated cannot-link pairs.
    """
    # Track newly created constraints
    new_must_link_count = 0
    new_cannot_link_count = 0

    print("Annotating Uncertain Positive Pairs (Within Clusters):")
    for i, j, dist in uncertain_positive_pairs:
        print(f"\nPair: ({i}, {j}), Distance: {dist:.4f}")
        print(f"Text 1: {all_texts[i]}")
        print(f"Text 2: {all_texts[j]}")
        while True:  # Keep asking for valid input
            decision = input("Should these belong in the same cluster? (y/n): ").strip().lower()
            if decision == "y":
                must_link_pairs.append((i, j))
                new_must_link_count += 1
                break
            elif decision == "n":
                cannot_link_pairs.append((i, j))
                new_cannot_link_count += 1
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    print("\nAnnotating Uncertain Negative Pairs (Across Clusters):")
    for i, j, dist in uncertain_negative_pairs:
        print(f"\nPair: ({i}, {j}), Distance: {dist:.4f}")
        print(f"Text 1: {all_texts[i]}")
        print(f"Text 2: {all_texts[j]}")
        while True:  # Keep asking for valid input
            decision = input("Should these belong in different clusters? (y/n): ").strip().lower()
            if decision == "y":
                cannot_link_pairs.append((i, j))
                new_cannot_link_count += 1
                break
            elif decision == "n":
                must_link_pairs.append((i, j))
                new_must_link_count += 1
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    # Display constraint updates
    print(f"\nCreated {new_must_link_count} must link constraints in this iteration.")
    print(f"Created {new_cannot_link_count} cannot link constraints in this iteration.")
    print(f"Total: {len(must_link_pairs)} must link constraints and {len(cannot_link_pairs)} cannot link constraints.")

    return must_link_pairs, cannot_link_pairs


def main():
    # Load the distance matrix
    try:
        distance_matrix = np.load("distance_matrix.npy")
        print("\nDistance matrix loaded successfully.")
    except FileNotFoundError:
        print("Error: 'distance_matrix.npy' file not found. Make sure the file is in the working directory.")
        exit()

    # Load the adjusted labels
    try:
        adjusted_labels = np.load("adjusted_labels.npy")
        print("Adjusted labels loaded successfully.")
    except FileNotFoundError:
        print("Error: 'adjusted_labels.npy' file not found. Make sure the file is in the working directory.")
        exit()

    # Load the sampled data
    try:
        sampled_data = pd.read_csv("sampled_data.csv")
        print("Sampled data loaded successfully.")
    except FileNotFoundError:
        print("Error: 'sampled_data.csv' file not found. Make sure the file is in the working directory.")
        exit()

    all_texts = sampled_data['TEXT'].tolist()

    # Initialize constraints
    must_link_pairs = []
    cannot_link_pairs = []

    # Select uncertain pairs
    uncertain_positive_pairs, uncertain_negative_pairs = select_uncertain_pairs(distance_matrix, adjusted_labels, must_link_pairs, cannot_link_pairs)

    # Display results
    print("\nUncertain Positive Pairs (within clusters):")
    for pair in uncertain_positive_pairs[:10]:
        print(f"Pair: {pair[0]} and {pair[1]}, Distance: {pair[2]:.4f}")

    print("\nUncertain Negative Pairs (across clusters):")
    for pair in uncertain_negative_pairs[:10]:
        print(f"Pair: {pair[0]} and {pair[1]}, Distance: {pair[2]:.4f}")

    # Annotate uncertain pairs
    must_link_pairs, cannot_link_pairs = annotate_and_update_constraints(
        uncertain_positive_pairs, uncertain_negative_pairs, all_texts, must_link_pairs, cannot_link_pairs
    )

    # Confirm updates
    print(f"\nUpdated Must-Link Pairs: {len(must_link_pairs)}")
    print(f"Updated Cannot-Link Pairs: {len(cannot_link_pairs)}")

    # Access the last 5 instances added to must_link_pairs and cannot_link_pairs
    print("\nMust-Link Pairs added:")
    for pair in must_link_pairs:  # [-5:]
        print(pair)

    print("Cannot-Link Pairs added:")
    for pair in cannot_link_pairs:  # [-5:]
        print(pair)  #

    # Save updated constraints
    np.save("must_link_pairs.npy", must_link_pairs)
    print("\nMust link pairs saved.")
    np.save("cannot_link_pairs.npy", cannot_link_pairs)
    print("Cannot link pairs saved.")


if __name__ == "__main__":
    main()
