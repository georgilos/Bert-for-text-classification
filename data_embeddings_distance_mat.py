import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


def generate_embeddings(texts, tokenizer, model, batch_size=32):

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
            # Extract [CLS] token embedding
            # `outputs.last_hidden_state` is a tensor of shape (batch_size, sequence_length, hidden_size)
            # We use the `[CLS]` token's embedding for each sentence (index 0 along sequence_length)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
        # print("Pre-normalization",torch.norm(batch_embeddings, p=2, dim=1))
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)  # L2 normalization
        # print("Post-normalization",torch.norm(batch_embeddings, p=2, dim=1))
        embeddings.append(batch_embeddings)

    # Stack all embeddings into a single tensor
    return torch.cat(embeddings, dim=0)


def main():

    # Load unlabeled data
    unlabeled_data = pd.read_csv('data/unlabeled_data/cleaned_texts_unlabeled_clear.csv', header=None, encoding='utf-8')

    # Because the .csv file has no headers, we must assign them
    unlabeled_data.columns = ['ID', 'TEXT']

    # Remove some non-printable characters that caused errors
    unlabeled_data['TEXT'] = unlabeled_data['TEXT'].fillna('').astype(str).str.replace(r'[\ufe0f\x0f\u0964]', '', regex=True)

    # Remove empty lines
    unlabeled_data = unlabeled_data[unlabeled_data['TEXT'].str.strip() != '']

    # Randomly select 100 rows
    sampled_data = unlabeled_data.sample(n=20, random_state=76)  # Number of samples & random_state

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Generate embeddings for the sampled data
    sampled_texts = sampled_data['TEXT'].tolist()
    sampled_embeddings = generate_embeddings(sampled_texts, tokenizer, model, batch_size=32)

    # Convert embeddings to NumPy array for the cdist() and compute pairwise distance matrix using cosine distance
    embeddings = sampled_embeddings.cpu().numpy()
    distance_matrix = cdist(embeddings, embeddings, metric='cosine')

    # Set diagonal to 0
    np.fill_diagonal(distance_matrix, 0)

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

    # Save the distance matrix
    np.save("distance_matrix.npy", distance_matrix)
    print("Distance matrix saved as 'distance_matrix.npy'")

    # Save the sampled embeddings
    torch.save(sampled_embeddings, "sampled_embeddings.pt")
    print("Sampled embeddings saved as 'sampled_embeddings.pt'")

    # Save sampled_data to a CSV
    sampled_data.to_csv("sampled_data.csv", index=False, encoding='utf-8-sig')
    print("Sampled data saved as 'sampled_data.csv'")

    # Amount of sampled data
    print(f"Created embeddings for {len(sampled_data)} lines")

    # First 5 instances of sampled data
    print(sampled_data.head(20))

    # The shape of the embeddings [#, 768]
    print("Sampled Embeddings Shape:", sampled_embeddings.shape)

    # The shape of the embeddings [#, #]
    print("Distance Matrix Shape:", distance_matrix.shape)

    # The mean distance of the embeddings
    print("Mean Distance in the distance matrix:", mean_distance)


if __name__ == "__main__":
    main()
