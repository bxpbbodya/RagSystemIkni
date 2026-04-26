from pathlib import Path
from core.index import build_faiss_index, load_chunks_from_jsonl

DATA_PATH = Path("data/local_cache.jsonl")

MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-small",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]

def main():
    chunks = load_chunks_from_jsonl(DATA_PATH)

    for model in MODELS:
        safe_name = model.replace("/", "_")

        print(f"\n🚀 Building index for {model}")

        build_faiss_index(
            chunks=chunks,
            embed_model_name=model,
            index_path=Path(f"data/index_{safe_name}.faiss"),
            meta_path=Path(f"data/index_{safe_name}_meta.jsonl"),
        )

if __name__ == "__main__":
    main()