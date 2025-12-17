import os
import glob
import pickle
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src import config

def load_data(embedding_model):
    if os.path.exists(config.EMBEDDINGS_PATH) and os.path.exists(config.CHUNKS_PATH):
        print(f"Cache found in '{config.CACHE_DIR}'. Loading...")
        try:
            doc_embeddings = np.load(config.EMBEDDINGS_PATH)
            with open(config.CHUNKS_PATH, 'rb') as f:
                all_chunks = pickle.load(f)
            return all_chunks, doc_embeddings
        except Exception as e:
            print(f"Cache error: {e}")

    print("Processing files...")
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    
    files = glob.glob(os.path.join(config.DATA_DIR, "*.txt"))
    if not files:
        files = glob.glob("*.txt") 
        
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    for file in files:
        file_name = os.path.basename(file)
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = text_splitter.create_documents([text])
        for chunk in chunks:
            chunk.metadata = {"source": file_name}
        all_chunks.extend(chunks)

    if not all_chunks:
        return [], None

    print("Computing embeddings...")
    doc_embeddings = embedding_model.encode([c.page_content for c in all_chunks], convert_to_tensor=True)
    doc_embeddings_np = doc_embeddings.cpu().numpy()

    np.save(config.EMBEDDINGS_PATH, doc_embeddings_np)
    with open(config.CHUNKS_PATH, 'wb') as f:
        pickle.dump(all_chunks, f)
        
    return all_chunks, doc_embeddings