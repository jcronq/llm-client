import json
import numpy as np
from llm_client.llm_utils import create_embedding_with_ada

import seaborn as sns
import matplotlib.pyplot as plt

def main():
    with open("embeddings.json", "r") as _f:
        embeddings = json.loads(_f.read())

    embeddings = [obj['query'] for obj in embeddings]

    # Calculate the dot product matrix
    dot_product_matrix = np.zeros((len(embeddings), len(embeddings)))

    for i, emb_i in enumerate(embeddings):
        for j, emb_j in enumerate(embeddings):
            dot_product_matrix[i][j] = np.dot(emb_i, emb_j)

    plt.figure(figsize=(10, 10))
    sns.heatmap(dot_product_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False)
    plt.xlabel("Query Index")
    plt.ylabel("Query Index")
    plt.title("Dot Product Matrix")
    plt.show()


    # print("query,result,dot")
    # for embedding in embeddings:
    #     print(np.dot(embedding["query"], embedding["response"]))


if __name__ == "__main__":
    main()
