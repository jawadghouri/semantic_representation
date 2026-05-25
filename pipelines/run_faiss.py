from pathlib import Path

from vectorstore.faiss_manager import FAISSManager

from utils.io_utils import load_numpy


EMBEDDING_DIR = Path(
    "data/processed/embeddings"
)

INDEX_DIR = Path(
    "data/processed/faiss"
)

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True
)


embedding_files = EMBEDDING_DIR.glob(
    "*.npy"
)


for file in embedding_files:

    print(f"\nProcessing {file.name}")

    embeddings = load_numpy(file)

    dimension = embeddings.shape[1]

    manager = FAISSManager(
        dimension
    )

    manager.add_embeddings(
        embeddings
    )

    index_name = (
        file.stem + ".index"
    )

    output_path = (
        INDEX_DIR / index_name
    )

    manager.save(
        str(output_path)
    )

    print(
        f"Saved -> {output_path}"
    )

print("\nAll indexes created.")