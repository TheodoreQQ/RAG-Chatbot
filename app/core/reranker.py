from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class Reranker:
  def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
    self.model = CrossEncoder(model_name)

  def rerank(self, query: str, docs: List[Document], top_n: int = 4) -> List[Document]:
    valid_docs = [doc for doc in docs if doc and doc.page_content.strip()]
    if not valid_docs:
      return []

    pairs = [[query, doc.page_content] for doc in valid_docs]
    scores = self.model.predict(pairs)

    doc_score_pairs = [(score, doc) for score, doc in zip(scores, valid_docs)]
    # Odrzucamy śmieciowe dopasowania (score <= 0.01)
    doc_score_pairs = [pair for pair in doc_score_pairs if pair[0] > 0.01]
    doc_score_pairs.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in doc_score_pairs[:top_n]]