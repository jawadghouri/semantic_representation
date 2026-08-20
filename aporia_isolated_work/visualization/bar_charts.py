"""
Prompt-response distance bar charts.

For each (LLM, embedder) combination, plots the Euclidean distances from the
prompt to each response. Shorter bars = response stayed closer to prompt topic.

Interpretation:
- Tight bars (low variance) = LLM consistently addresses the prompt
- Wide range = LLM diverges in some responses
- Compare across LLMs for each embedder to see which LLM stays on-topic best
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from typing import Dict, List, Optional

from config.groups import EMBEDDING_MODELS, DATA_DIR, OUTPUT_DIR


def load_embedding(
    data_dir: str,
    filename: str,
    normalize: bool = False
) -> Optional[np.ndarray]:
    """Load a single .npy embedding file."""
    fpath = Path(data_dir) / filename
    if fpath.exists():
        try:
            vec = np.load(fpath)
            if vec.ndim > 1:
                vec = vec.flatten()
            if normalize and np.linalg.norm(vec) > 0:
                vec = vec / np.linalg.norm(vec)
            return vec
        except Exception as e:
            print(f"Warning: error loading {filename}: {e}")
    else:
        print(f"Warning: {fpath} not found")
    return None


def compute_distances(
    prompt_vec: np.ndarray,
    response_vecs: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Compute Euclidean distance from prompt to each response.

    Args:
        prompt_vec: single prompt embedding
        response_vecs: dict of response_id -> embedding

    Returns:
        dict of response_id -> distance
    """
    distances = {}
    for rid, vec in response_vecs.items():
        dist = np.linalg.norm(prompt_vec - vec)
        distances[rid] = dist
    return distances


def plot_prompt_response_distances(
    prompt_vec: np.ndarray,
    response_vecs: Dict[str, np.ndarray],
    response_ids: List[str],
    title: str,
    output_path: str,
    figsize: tuple = (12, 6)
):
    """
    Plot bar chart of prompt-response distances.

    Args:
        prompt_vec: single prompt embedding
        response_vecs: dict of response_id -> embedding
        response_ids: ordered list of response IDs for x-axis
        title: chart title
        output_path: where to save PNG
        figsize: (width, height) for figure
    """
    distances = compute_distances(prompt_vec, response_vecs)

    ordered_ids = [rid for rid in response_ids if rid in distances]
    ordered_dists = [distances[rid] for rid in ordered_ids]

    n = len(ordered_ids)
    indices = np.arange(n)

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        indices, ordered_dists,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.8
    )

    mean_dist = np.mean(ordered_dists)
    ax.axhline(
        mean_dist,
        color="tomato",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_dist:.3f}"
    )

    min_idx = np.argmin(ordered_dists)
    max_idx = np.argmax(ordered_dists)
    bars[min_idx].set_color("lightgreen")
    bars[min_idx].set_edgecolor("darkgreen")
    bars[min_idx].set_linewidth(1)
    bars[max_idx].set_color("lightcoral")
    bars[max_idx].set_edgecolor("darkred")
    bars[max_idx].set_linewidth(1)

    ax.set_xlabel("Response ID", fontsize=11)
    ax.set_ylabel("Euclidean Distance from Prompt", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(indices)
    ax.set_xticklabels(ordered_ids, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()

    return {
        "mean": mean_dist,
        "std": float(np.std(ordered_dists)),
        "min": float(np.min(ordered_dists)),
        "max": float(np.max(ordered_dists)),
        "n": n
    }


def plot_all_combinations(
    data_dir: str = DATA_DIR,
    output_dir: str = OUTPUT_DIR,
    models: List[str] = EMBEDDING_MODELS,
    llms: List[str] = ["llama", "mistral", "phi"],
    prompt_id: str = "prompt",
    normalized: bool = False
):
    """
    Generate bar charts for all (LLM, embedder) combinations.

    Args:
        data_dir: directory containing .npy embedding files
        output_dir: directory for output PNG files
        models: embedding model names
        llms: LLM names
        prompt_id: prefix for prompt embedding file (e.g. "prompt" -> "prompt_bge.npy")
        normalized: whether to load normalized embeddings
    """
    os.makedirs(output_dir, exist_ok=True)

    suffix_text = "_norm" if normalized else ""

    results = {}

    for model_name in models:
        prompt_filename = f"{prompt_id}_{model_name}{suffix_text}.npy"
        prompt_vec = load_embedding(data_dir, prompt_filename, normalize=False)

        if prompt_vec is None:
            print(f"⚠ Skipping {model_name} — prompt file not found")
            continue

        for llm_name in llms:
            response_vecs = {}
            response_ids = []

            idx = 1
            while True:
                response_filename = f"{llm_name}_response_{idx}_{model_name}{suffix_text}.npy"
                response_path = Path(data_dir) / response_filename

                if not response_path.exists():
                    break

                vec = load_embedding(data_dir, response_filename, normalize=False)
                if vec is not None:
                    response_vecs[f"R{idx}"] = vec
                    response_ids.append(f"R{idx}")

                idx += 1

            if not response_vecs:
                print(f"⚠ No responses found for {llm_name} + {model_name}")
                continue

            title = f"{llm_name.upper()} → {model_name.upper()} (Prompt Proximity)\n" \
                    f"Response distances from prompt embedding"

            output_filename = f"{llm_name}_{model_name}_prompt_response{suffix_text}.png"
            output_path = os.path.join(output_dir, output_filename)

            stats = plot_prompt_response_distances(
                prompt_vec,
                response_vecs,
                response_ids,
                title,
                output_path
            )

            results[f"{llm_name}_{model_name}"] = stats

    print(f"\n✓ Generated charts for {len(results)} LLM+embedder combinations")
    return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PROMPT-RESPONSE DISTANCE BAR CHARTS")
    print("=" * 70)

    print("\n[1/2] Plotting UNNORMALIZED prompt-response distances...")
    plot_all_combinations(normalized=False)

    print("\n[2/2] Plotting NORMALIZED prompt-response distances...")
    plot_all_combinations(normalized=True)

    print("\n✓ All bar charts generated successfully!")
    print(f"✓ Check '{OUTPUT_DIR}/' for PNG files")
