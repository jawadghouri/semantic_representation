import json


def evaluate_retrieval(
        query,
        results
):

    print("\nQUERY\n")
    print(query)

    print("\nTOP RESULTS\n")

    for idx, item in enumerate(results):

        print(
            f"\nResult {idx+1}"
        )

        print(
            item["response"]
        )