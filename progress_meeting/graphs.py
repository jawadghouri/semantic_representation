##
# This generates line graphs for response centric comparisons. 
# It handles both normalized and unnormalized embeddings, 
# saving the resulting plots to specified directories.
##

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def plot_embedding_comparisons(data_dir: str, output_dir: str, ids: list, normalized: bool):
    """
    Loads embedding files for given IDs and plots them on line graphs.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    dir_path = Path(data_dir)

    # The styles you requested
    styles = {
        "minilm": {"linestyle": "-", "label": "MiniLM"},
        "bge": {"linestyle": ":", "label": "BGE"},
        "e5": {"linestyle": "-.", "label": "E5"}
    }

    for current_id in ids:
        plt.figure(figsize=(12, 6))
        
        # Track if we found any files for this ID
        found_data = False 

        for embed_name, style_params in styles.items():
            if normalized == True:
                file_name = f"{current_id}_{embed_name}_norm.npy"
            else:
                file_name = f"{current_id}_{embed_name}.npy"
            file_path = dir_path / file_name

            if file_path.exists():
                try:
                    # Load the 1D array
                    data = np.load(file_path)
                    
                    # Ensure it's 1D for plotting
                    if data.ndim > 1:
                        data = data.flatten()

                    # Plot the line
                    plt.plot(
                        data, 
                        linestyle=style_params["linestyle"], 
                        label=f"{style_params['label']} ({len(data)} dims)",
                        alpha=0.7 # Slight transparency to see overlapping lines
                    )
                    found_data = True
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
            else:
                print(f"Warning: Could not find {file_path}")

        if found_data:
            # Formatting the graph
            plt.title(f"Embedding Vector Values for Response ID: {current_id}", fontsize=14)
            if normalized:
                plt.xlabel("Vector Dimension Index (Normalized)", fontsize=12)
            else:
                plt.xlabel("Vector Dimension Index (Unnormalized)", fontsize=12)
            plt.ylabel("Embedding Value", fontsize=12)
            plt.legend(loc="upper right")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            # Save the plot
            save_path = os.path.join(output_dir, f"{current_id}_comparison.png")
            plt.savefig(save_path, dpi=300)
            print(f"Saved plot: {save_path}")
        else:
            print(f"Skipping {current_id} - no data found.")
            
        plt.close() # Close figure to free memory

if __name__ == "__main__":
    # Point these to your actual directories
    EMBEDDINGS_FOLDER = "progress_meeting/embeddings"
    PLOTS_FOLDER = "progress_meeting/plots_response_centric"
    
    # List the IDs you want to graph
    my_ids = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]
    my_models = ["minilm", "bge", "e5"]
    plot_embedding_comparisons(EMBEDDINGS_FOLDER, PLOTS_FOLDER, my_ids, normalized=False)

    print("\nNow plotting normalized embeddings...\n")
     # Point these to your actual directories
    EMBEDDINGS_FOLDER = "progress_meeting/embeddings_norm"
    PLOTS_FOLDER = "progress_meeting/plots_response_centric_norm"
    
    # List the IDs you want to graph
    my_ids = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]
    my_models = ["minilm", "bge", "e5"]
    plot_embedding_comparisons(EMBEDDINGS_FOLDER, PLOTS_FOLDER, my_ids, normalized=True)