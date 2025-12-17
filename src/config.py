import os

DATA_DIR = "data"
CACHE_DIR = "cache_store"
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "embeddings.npy")
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.pkl")

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
LLM_MODEL_NAME = "groq/llama-3.1-8b-instant"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200