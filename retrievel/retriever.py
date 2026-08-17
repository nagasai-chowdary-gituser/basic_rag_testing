from embeddings.embedder import Embedder
from vectorstore.store import search
from config.settings import TOP_K

class Retriever:
    def __init__(self):
        self.embedder=Embedder()
    def retrieve(self,query,top_k=TOP_K):
        if not query or not query.strip():
            raise ValueError("query must be filled")
        query_embedding=self.embedder.embed_query(query)
        results=search(query_embedding=query_embedding,top_k=top_k)
        return self._format_results(results)
    def _format_results(self,results):
        documents=results.get("documents",[[]])[0]
        metadatas=results.get("metadatas",[[]])[0]
        distances=results.get("distances",[[]])[0]
        retrieved_chunks=[]
        for document,metadata,distance in zip(documents,metadatas,distances):
            retrieved_chunks.append({"text":document,"metadata":metadata,"distance":distance})
        return retrieved_chunks