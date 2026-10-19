from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
        valid_docs = [doc for doc in docs if doc and doc.page_content.strip()]
        if not valid_docs:
            return []

        pairs = [[query, doc.page_content] for doc in valid_docs]
        scores = self.model.predict(pairs)

        # 1. Łączymy w pary i od razu sortujemy od najlepszego wyniku do najgorszego
        doc_score_pairs = [(score, doc) for score, doc in zip(scores, valid_docs)]
        doc_score_pairs.sort(key=lambda x: x[0], reverse=True)

        # 2. Pobieramy wynik najlepszego dokumentu (lidera)
        best_score = doc_score_pairs[0][0]
        
        print(f"\n--- DEBUG RERANKERA ---")

        # 3. Odsiew relatywny: bierzemy maksymalnie top_n dokumentów, 
        # ale odrzucamy te, które są słabsze od lidera o więcej niż 4 punkty.
        min_score = 0.05
        
        filtered_docs = []
        for score, doc in doc_score_pairs[:top_n]:
            
            print(f"Score: {score:.2f} | Różnica do lidera: {best_score - score:.2f}")
            
            if score >= best_score - 0.3 and score >= min_score:
                filtered_docs.append(doc)
        
        print(f"Odrzucono dokumentów przez próg relatywny: {len(doc_score_pairs[:top_n]) - len(filtered_docs)}")
        print("------------------------------------------------\n")
        
        return filtered_docs