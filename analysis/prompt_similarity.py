import numpy as np


def euclidean_distance(vec1, vec2):
    """
    Compute Euclidean distance between two unnormalized embeddings.
    """
    return float(np.linalg.norm(vec1 - vec2))


def compare_prompt_to_responses(prompt_embedding,
                                response_embeddings):
    """
    Parameters
    ----------
    prompt_embedding : np.ndarray

    response_embeddings : dict
        {
            "llama": embedding,
            "mistral": embedding,
            ...
        }

    Returns
    -------
    list
    """

    results = []

    for model_name, embedding in response_embeddings.items():

        distance = euclidean_distance(
            prompt_embedding,
            embedding
        )

        results.append({
            "model": model_name,
            "distance": distance
        })

    results.sort(
        key=lambda x: x["distance"]
    )

    return results