##
# This generates line graphs for model centric comparisons.
# It handles both normalized and unnormalized embeddings,
# saving the resulting plots to specified directories.
##

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def plot_model_centric_comparisons(data_dir: str, output_dir: str, models: list, ids: list, normalized: bool):
    """
    Loads embedding files grouped by model and plots all IDs on the same graph.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    dir_path = Path(data_dir)

    for embed_name in models:
        # Create a new figure for each model
        plt.figure(figsize=(12, 6))
        
        found_data = False 

        for current_id in ids:
            if normalized:
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

                    # Plot the line for this specific response ID
                    plt.plot(
                        data, 
                        label=current_id,
                        linewidth=1.2,
                        alpha=0.6 # High transparency so overlapping lines blend
                    )
                    found_data = True
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
            else:
                print(f"Warning: Could not find {file_path}")

        if found_data:
            # Formatting the graph
            plt.title(f"Embedding Vector Variations across Responses\nModel: {embed_name.upper()}", fontsize=14)
            if normalized:
                plt.xlabel("Vector Dimension Index (Normalized)", fontsize=12)
            else:
                plt.xlabel("Vector Dimension Index (Unnormalized)", fontsize=12)
            plt.ylabel("Embedding Value", fontsize=12)
            
            # Place the legend outside the plot area so it doesn't cover the 9 lines
            plt.legend(title="Response IDs", bbox_to_anchor=(1.02, 1), loc="upper left")
            
            plt.grid(True, linestyle="--", alpha=0.5)
            
            # Adjust layout to make room for the external legend
            plt.tight_layout()

            # Save the plot
            save_path = os.path.join(output_dir, f"{embed_name}_comparison.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {save_path}")
        else:
            print(f"Skipping {embed_name} - no data found.")
            
        plt.close() # Close figure to free memory

if __name__ == "__main__":
    # Point these to your actual directories
    EMBEDDINGS_FOLDER = "progress_meeting/embeddings"
    PLOTS_FOLDER = "progress_meeting/plots_model_centric"
    
    # The models you want a graph for
    my_models = ["minilm", "bge", "e5"]
    
    # The responses you want plotted as lines on those graphs
    my_ids = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]
    
    plot_model_centric_comparisons(EMBEDDINGS_FOLDER, PLOTS_FOLDER, my_models, my_ids, normalized=False)


    print("\nNow plotting normalized embeddings...\n")


    # Point these to your actual directories
    EMBEDDINGS_FOLDER = "progress_meeting/embeddings_norm"
    PLOTS_FOLDER = "progress_meeting/plots_model_centric_norm"
    
    # The models you want a graph for
    my_models = ["minilm", "bge", "e5"]
    
    # The responses you want plotted as lines on those graphs
    my_ids = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]
    
    plot_model_centric_comparisons(EMBEDDINGS_FOLDER, PLOTS_FOLDER, my_models, my_ids, normalized=True)