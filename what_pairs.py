import numpy as np
import pandas as pd

# Load the pairs
must_link_pairs = np.load("must_link_pairs.npy", allow_pickle=True)
cannot_link_pairs = np.load("cannot_link_pairs.npy", allow_pickle=True)

# Load the sampled data to get the texts
sampled_data = pd.read_csv("sampled_data.csv")
all_texts = sampled_data['TEXT'].tolist()

# Function to display the texts for a list of pairs
def display_pair_texts(pairs, all_texts, pair_type):
    print(f"\n{pair_type} Pairs:")
    for pair in pairs:
        index1, index2 = pair
        text1 = all_texts[index1]
        text2 = all_texts[index2]
        print(f"Pair: ({index1}, {index2})")
        print(f"Text 1: {text1}")
        print(f"Text 2: {text2}")
        print("-" * 20)  # Separator between pairs


# Display the texts for must-link pairs
display_pair_texts(must_link_pairs, all_texts, "Must-Link")

# Display the texts for cannot-link pairs
display_pair_texts(cannot_link_pairs, all_texts, "Cannot-Link")