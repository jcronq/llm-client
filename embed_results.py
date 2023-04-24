import json

from llm_client.llm_utils import create_embedding_with_ada


def main():
    with open("results.json", "r") as _f:
        results = json.loads(_f.read())

    embeddings = []
    for result in results:
        embeddings.append(
            {
                "query": create_embedding_with_ada(result["query"]),
                "response": create_embedding_with_ada(result["response"]),
            }
        )

    with open("embeddings.json", "w") as _f:
        _f.write(json.dumps(embeddings))


if __name__ == "__main__":
    main()
