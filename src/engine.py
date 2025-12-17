import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from src import config
from src.data_manager import load_data

class HybridRetriever:
    def __init__(self):
        print("Initializing Retrieval Engine...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.reranker = CrossEncoder(config.RERANKER_MODEL_NAME)
        
        self.documents, self.doc_embeddings = load_data(self.embedding_model)
        
        if self.documents:
            self.texts = [doc.page_content for doc in self.documents]
            print("Building BM25 index...")
            tokenized_corpus = [doc.split(" ") for doc in self.texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            print("No documents loaded.")

    def search(self, query, mode="Hybrid", top_k=3):
        if not self.documents:
            return []
            
        initial_top_k = 30 
        
        bm25_scores = self.bm25.get_scores(query.split(" "))
        
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        semantic_results = util.semantic_search(query_embedding, self.doc_embeddings, top_k=len(self.documents))[0]
        
        semantic_scores = np.zeros(len(self.documents))
        for res in semantic_results:
            semantic_scores[res['corpus_id']] = res['score']

        if mode == "Keyword (BM25)":
            final_scores = bm25_scores
        elif mode == "Semantic (Vectors)":
            final_scores = semantic_scores
        else:
            bm25_ranks = np.argsort(bm25_scores)[::-1]
            semantic_ranks = np.argsort(semantic_scores)[::-1]
            rrf_scores = np.zeros(len(self.documents))
            k = 60
            for rank, doc_idx in enumerate(bm25_ranks):
                rrf_scores[doc_idx] += 1 / (k + rank)
            for rank, doc_idx in enumerate(semantic_ranks):
                rrf_scores[doc_idx] += 1 / (k + rank)
            final_scores = rrf_scores

        top_indices = np.argsort(final_scores)[::-1][:initial_top_k]
        candidate_docs = [self.documents[idx] for idx in top_indices]

        pairs = [[query, doc.page_content] for doc in candidate_docs]
        rerank_scores = self.reranker.predict(pairs)
        
        reranked_indices = np.argsort(rerank_scores)[::-1][:top_k]
        
        results = []
        for idx in reranked_indices:
            results.append((candidate_docs[idx], rerank_scores[idx]))
            
        return results