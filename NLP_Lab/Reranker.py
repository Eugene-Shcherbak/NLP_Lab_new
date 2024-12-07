from rerankers import Reranker

ranker = Reranker('cross-encoder')


def rerank(docs: [str], query: str):
    results = ranker.rank(query, docs)
    return results
