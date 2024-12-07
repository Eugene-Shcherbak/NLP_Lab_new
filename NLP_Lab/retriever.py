import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from chunking import chunk_documents
import numpy as np


class Retriever:

    def __init__(self, docs: [str]) -> None:
        self.docs = chunk_documents(docs)

        tokenized_docs = [doc.lower().split(" ") for doc in self.docs]

        self.bm25 = BM25Okapi(tokenized_docs)
        self.sbert = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

        self.doc_embeddings = self.sbert.encode(self.docs)

    def get_docs(self, query, methods, n=20) -> [str]:
        bm25_scores = self._get_bm25_scores(query) if "BM25" in methods else torch.zeros(len(self.docs))
        semantic_scores = self._get_semantic_scores(query) if "semantic" in methods else torch.zeros(len(self.docs))

        scores = 0.3 * bm25_scores + 0.7 * semantic_scores

        sorted_indices = scores.argsort(descending=True)
        result = [self.docs[i] for i in sorted_indices[:n]]
        return result

    def _get_bm25_scores(self, query):
        tokenized_query = query.lower().split(" ")
        return torch.tensor(self.bm25.get_scores(tokenized_query))

    def _get_semantic_scores(self, query):
        query_embedding = self.sbert.encode(query)
        scores = np.dot(self.doc_embeddings, query_embedding) / (
                np.linalg.norm(self.doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        return torch.tensor(scores)
