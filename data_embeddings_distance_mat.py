import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np


def generate_embeddings(texts, tokenizer, model, batch_size=16, use_cls=True):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)  # Ensure the model is on the correct device
    embeddings = []
    for i in range(0, len(texts), batch_size):  # Create batches
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)

        # Move all input tensors to the same device as the model
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        if use_cls:
            # Extract [CLS] token embedding
            # `outputs.last_hidden_state` is a tensor of shape (batch_size, sequence_length, hidden_size)
            # We use the `[CLS]` token's embedding for each sentence (index 0 along sequence_length)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            # Apply mean pooling
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)

        embeddings.append(batch_embeddings)

    # Stack all embeddings into a single tensor
    return torch.cat(embeddings, dim=0)


def main():

    # Load unlabeled data
    unlabeled_data = pd.read_csv('data/unlabeled_data/cleaned_texts_unlabeled.csv', header=None, encoding='utf-8')
    # Because the .csv file has no headers, we must assign them
    unlabeled_data.columns = ['ID', 'TEXT']
    # Randomly select 100 rows
    sampled_data = unlabeled_data.sample(n=300, random_state=21)  # Number of samples & random_state

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Generate embeddings for the sampled data
    sampled_texts = sampled_data['TEXT'].tolist()
    sampled_embeddings = generate_embeddings(sampled_texts, tokenizer, model, batch_size=16, use_cls=True)

    # Convert embeddings to NumPy array and compute pairwise distance matrix using cosine distance
    embeddings = sampled_embeddings.cpu().numpy()
    distance_matrix = cdist(embeddings, embeddings, metric='cosine')

    # Compute mean distance
    mean_distance = np.mean(distance_matrix)

    # Visualize the distances
    plt.hist(distance_matrix.flatten(), bins=50)
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.title("Distribution of Pairwise Distances")

    # Saving an image of the plot for visualization
    plt.savefig("pairwise_distances_histogram.png")
    print("Plot of distances saved as 'pairwise_distances_histogram.png'")

    # Saving the files that will be used later
    # Save the distance matrix
    np.save("distance_matrix.npy", distance_matrix)
    print("Distance matrix saved as 'distance_matrix.npy'")
    # Save the sampled embeddings
    torch.save(sampled_embeddings, "sampled_embeddings.pt")
    print("Sampled embeddings saved as 'sampled_embeddings.pt'")
    # Save sampled_data to a CSV

    sampled_data.to_csv("sampled_data.csv", index=False, encoding='utf-8-sig')
    print("Sampled data saved as 'sampled_data.csv'")

    # Print statements
    # Amount of sampled data
    print(f"Created embeddings for {len(sampled_data)} lines")
    # First 5 instances of sampled data
    print(sampled_data.head(15))
    # The shape of the embeddings [#, 768]
    print("Sampled Embeddings Shape:", sampled_embeddings.shape)
    # The shape of the embeddings [#, #]
    print("Distance Matrix Shape:", distance_matrix.shape)
    # The mean distance of the embeddings
    print("Mean Distance in the distance matrix:", mean_distance)


if __name__ == "__main__":
    main()
