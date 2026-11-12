import chromadb
from chromadb.utils import embedding_functions
import hashlib

class SemanticCache:
    """Lokalna, semantyczna pamięć podręczna (Cache) oparta na wektorach."""
    
    def __init__(self, threshold=0.35, cache_dir="./chroma_cache"):
        self.client = chromadb.PersistentClient(path=cache_dir)
        self.emb_fn = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name="query_cache", 
            embedding_function=self.emb_fn
        )
        self.threshold = threshold

    def search(self, query: str):
        """Szuka semantycznie podobnego pytania w pamięci."""
        results = self.collection.query(query_texts=[query], n_results=1)
        
        if results['distances'] and results['distances'][0]:
            distance = results['distances'][0][0]
            print(f"[DEBUG CACHE] Odległość semantyczna: {distance:.4f} (Próg: {self.threshold})", flush=True)
            
            if distance < self.threshold:
                # zwracana zostaje gotowa odpowiedź wyciągnięta z metadanych
                return results['metadatas'][0][0]['answer']
        return None

    def save(self, query: str, response: str):
        """Zapisuje parę pytanie-odpowiedź do pamięci wektorowej."""
        doc_id = hashlib.md5(query.encode('utf-8')).hexdigest()
        
        self.collection.upsert(
            documents=[query],                
            metadatas=[{"answer": response}], 
            ids=[doc_id]
        )
    
    def clear_cache(self):
        # po wgraniu nowych plików usuwa całą pamięć podręczną
        try:
            self.client.delete_collection("query_cache")
        except ValueError:
            pass
        
        self.collection = self.client.create_collection(
            name="query_cache",
            embedding_function=self.emb_fn
        )
        print("Cache został wyczyszczony z nieaktualnych dokumentów", flush=True)