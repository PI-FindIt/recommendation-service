# src/models/recommendation_engine.py
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class RecommendationEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2') # all-mpnet-base-v2
        self.index = None
        self.product_mapping = {}
        
    async def get_recommendations(
        self,
        user_id: str,
        n_recommendations: int = 5
    ) -> List[dict]:
        # Implementação básica inicial
        pass

    def _compute_product_embeddings(self, product_descriptions: List[str]):
        return self.model.encode(product_descriptions)

    def _build_faiss_index(self, embeddings: np.ndarray):
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)