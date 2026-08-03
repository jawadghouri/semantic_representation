import json


def save_report(
        statistics,
        labels,
        distance_matrix,
        output_file
):

    report = {

        "models":
            labels,

        "statistics":
            statistics,

        "distance_matrix":
            distance_matrix.tolist()
    }

    with open(
            output_file,
            "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )