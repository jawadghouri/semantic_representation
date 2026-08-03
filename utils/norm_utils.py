import numpy as np
from pathlib import Path

def check_embeddings_normalization(directory_path: str, tolerance: float = 1e-4):
    """
    Scans a directory for .npy files and checks if the stored embeddings are L2 normalized.
    """
    dir_path = Path(directory_path)
    
    if not dir_path.is_dir():
        print(f"Directory not found: {directory_path}")
        return

    npy_files = list(dir_path.glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in the directory: {directory_path}")
        return

    print(f"Analyzing {len(npy_files)} files in '{directory_path}'...\n")
    
    # Updated column header
    print(f"{'Filename':<30} | {'Status':<16} | {'L2 Norm':<12}")
    print("-" * 65)

    for file_path in sorted(npy_files):
        try:
            # Load the numpy array
            data = np.load(file_path)

            # Handle empty files
            if data.size == 0:
                print(f"{file_path.name:<30} | {'EMPTY':<16} | N/A")
                continue

            # Ensure data is 2D (batch_size, embedding_dim)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            elif data.ndim > 2:
                print(f"{file_path.name:<30} | {'INVALID SHAPE':<16} | N/A (Shape: {data.shape})")
                continue

            # Calculate L2 norms along the embedding axis
            norms = np.linalg.norm(data, axis=1)

            # Check if all vectors are close to 1 within the given tolerance
            is_normalized = np.allclose(norms, 1.0, atol=tolerance)
            
            # Extract the exact L2 norm for the single vector
            if len(norms) == 1:
                l2_norm_display = f"{norms[0]:.6f}"
            else:
                l2_norm_display = f"{np.mean(norms):.6f} (mean)"

            if is_normalized:
                status = "❌ NORMALIZED"
            else:
                status = "✅ UNNORMALIZED"

            print(f"{file_path.name:<30} | {status:<16} | {l2_norm_display}")

        except Exception as e:
            print(f"{file_path.name:<30} | {'ERROR':<16} | {str(e)}")

    print("-" * 65)

if __name__ == "__main__":
    # Point this to the directory where you are saving your arrays
    target_directory = "./embeddings_output" 
    check_embeddings_normalization(target_directory)

# # import numpy as np
# # from pathlib import Path

# # def check_embeddings_normalization(directory_path: str, tolerance: float = 1e-4):
# #     """
# #     Scans a directory for .npy files and checks if the stored embeddings are L2 normalized.
# #     """
# #     dir_path = Path(directory_path)
    
# #     if not dir_path.is_dir():
# #         print(f"Directory not found: {directory_path}")
# #         return

# #     npy_files = list(dir_path.glob("*.npy"))
    
# #     if not npy_files:
# #         print(f"No .npy files found in the directory: {directory_path}")
# #         return

# #     print(f"Analyzing {len(npy_files)} files in '{directory_path}'...\n")
# #     print(f"{'Filename':<30} | {'Status':<16} | {'Mean L2 Norm':<12}")
# #     print("-" * 65)

# #     for file_path in sorted(npy_files):
# #         try:
# #             # Load the numpy array
# #             data = np.load(file_path)

# #             # Handle empty files
# #             if data.size == 0:
# #                 print(f"{file_path.name:<30} | {'EMPTY':<16} | N/A")
# #                 continue

# #             # Ensure data is 2D (batch_size, embedding_dim)
# #             if data.ndim == 1:
# #                 data = data.reshape(1, -1)
# #             elif data.ndim > 2:
# #                 print(f"{file_path.name:<30} | {'INVALID SHAPE':<16} | N/A (Shape: {data.shape})")
# #                 continue

# #             # Calculate L2 norms along the embedding axis
# #             norms = np.linalg.norm(data, axis=1)

# #             # Check if all vectors are close to 1 within the given tolerance
# #             is_normalized = np.allclose(norms, 1.0, atol=tolerance)
# #             mean_norm = np.mean(norms)

# #             if is_normalized:
# #                 status = "❌ NORMALIZED"
# #             else:
# #                 status = "✅ UNNORMALIZED"

# #             print(f"{file_path.name:<30} | {status:<16} | {mean_norm:.6f}")

# #         except Exception as e:
# #             print(f"{file_path.name:<30} | {'ERROR':<16} | {str(e)}")

# #     print("-" * 65)

# # if __name__ == "__main__":
# #     # Point this to the directory where you are saving your arrays
# #     target_directory = "./embeddings_output" 
# #     check_embeddings_normalization(target_directory)