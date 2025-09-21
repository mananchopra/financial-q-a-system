"""Vector store implementation using ChromaDB for financial documents."""
import uuid
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from rich.console import Console
from rich.progress import Progress

console = Console()

class FinancialVectorStore:
    """ChromaDB-based vector store for financial document chunks."""
    
    def __init__(self, persist_directory: str, collection_name: str, google_api_key: str, embedding_model: str = "models/text-embedding-004"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        genai.configure(api_key=google_api_key)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        try:
            self.collection = self.client.get_collection(name=collection_name)
            console.print(f"[green]Loaded existing collection: {collection_name}[/green]")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Financial 10-K filing chunks"}
            )
            console.print(f"[blue]Created new collection: {collection_name}[/blue]")
    
    def add_documents(self, chunks: List[Dict]) -> None:
        """Add document chunks to the vector store."""
        if not chunks:
            console.print("[yellow]No chunks to add[/yellow]")
            return
        
        console.print(f"[blue]Adding {len(chunks)} chunks to vector store...[/blue]")
        
        batch_size = 50
        
        with Progress() as progress:
            task = progress.add_task("Processing chunks...", total=len(chunks))
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                self._process_batch(batch)
                progress.advance(task, len(batch))
        
        console.print(f"[green]Successfully added {len(chunks)} chunks to vector store[/green]")
    
    def _process_batch(self, batch: List[Dict]) -> None:
        """Process a batch of chunks."""
        texts = [chunk["text"] for chunk in batch]
        embeddings = self._get_embeddings(texts)
        ids = []
        documents = []
        metadatas = []
        
        for chunk, embedding in zip(batch, embeddings):
            doc_id = chunk.get("chunk_id", str(uuid.uuid4()))
            ids.append(doc_id)
            documents.append(chunk["text"])
            metadata = {
                "company": chunk["company"],
                "year": str(chunk["year"]),
                "section": chunk["section"],
                "chunk_number": chunk.get("chunk_number", 0),
                "token_count": chunk.get("token_count", 0),
                "financial_score": chunk.get("financial_score", 0),
                "has_financial_data": chunk.get("metadata", {}).get("has_financial_data", False)
            }
            metadatas.append(metadata)
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = []
            for text in texts:
                # Gemini API handles one text at a time
                response = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(response['embedding'])
            return embeddings
        except Exception as e:
            console.print(f"[red]Error generating embeddings: {e}[/red]")
            raise
    
    def search(self, query: str, n_results: int = 8, filters: Optional[Dict] = None) -> List[Dict]:
        """Search for relevant chunks based on query."""
        # Generate query embedding
        try:
            response = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = response['embedding']
        except Exception as e:
            console.print(f"[red]Error generating query embedding: {e}[/red]")
            raise
        
        # Prepare where clause for filtering
        where_clause = {}
        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    where_clause[key] = {"$in": value}
                else:
                    where_clause[key] = value
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause if where_clause else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "company": results["metadatas"][0][i]["company"],
                "year": int(results["metadatas"][0][i]["year"]),
                "section": results["metadatas"][0][i]["section"]
            })
        
        return formatted_results
    
    def search_by_company_year(self, query: str, company: Optional[str] = None, year: Optional[int] = None, n_results: int = 5) -> List[Dict]:
        """Search with company and year filters."""
        filters = {}
        if company:
            filters["company"] = company
        if year:
            filters["year"] = str(year)
        
        return self.search(query, n_results, filters)
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        count = self.collection.count()
        
        # Get sample of metadata to analyze
        sample_results = self.collection.get(limit=min(100, count), include=["metadatas"])
        
        companies = set()
        years = set()
        sections = set()
        
        for metadata in sample_results["metadatas"]:
            companies.add(metadata["company"])
            years.add(metadata["year"])
            sections.add(metadata["section"])
        
        return {
            "total_chunks": count,
            "companies": sorted(list(companies)),
            "years": sorted(list(years)),
            "sections": sorted(list(sections))
        }
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate collection
        try:
            self.client.delete_collection(name=self.collection_name)
            console.print(f"[yellow]Deleted collection: {self.collection_name}[/yellow]")
        except Exception:
            pass
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Financial 10-K filing chunks"}
        )
        console.print(f"[blue]Created new empty collection: {self.collection_name}[/blue]")

class RetrievalEngine:
    """Advanced retrieval engine with multiple search strategies."""
    
    def __init__(self, vector_store: FinancialVectorStore):
        self.vector_store = vector_store
    
    def retrieve_for_query(self, query: str, strategy: str = "semantic", **kwargs) -> List[Dict]:
        """Retrieve relevant chunks using different strategies."""
        if strategy == "semantic":
            return self._semantic_search(query, **kwargs)
        elif strategy == "hybrid":
            return self._hybrid_search(query, **kwargs)
        elif strategy == "company_focused":
            return self._company_focused_search(query, **kwargs)
        elif strategy == "temporal":
            return self._temporal_search(query, **kwargs)
        else:
            return self._semantic_search(query, **kwargs)
    
    def _semantic_search(self, query: str, n_results: int = 8, **kwargs) -> List[Dict]:
        """Basic semantic similarity search."""
        return self.vector_store.search(query, n_results)
    
    def _hybrid_search(self, query: str, n_results: int = 8, **kwargs) -> List[Dict]:
        """Hybrid search combining semantic and keyword matching."""
        # Get semantic results
        semantic_results = self.vector_store.search(query, n_results)
        
        # Simple keyword boost for financial terms
        financial_keywords = ["revenue", "income", "profit", "margin", "earnings", "sales"]
        query_lower = query.lower()
        
        for result in semantic_results:
            # Boost score if text contains query keywords
            text_lower = result["text"].lower()
            keyword_matches = sum(1 for keyword in financial_keywords if keyword in query_lower and keyword in text_lower)
            
            # Adjust distance (lower is better)
            if keyword_matches > 0:
                result["distance"] = result["distance"] * (1 - 0.1 * keyword_matches)
                result["keyword_boost"] = keyword_matches
        
        # Re-sort by adjusted distance
        semantic_results.sort(key=lambda x: x["distance"])
        
        return semantic_results
    
    def _company_focused_search(self, query: str, companies: List[str], n_results: int = 8, **kwargs) -> List[Dict]:
        """Search focused on specific companies."""
        all_results = []
        
        for company in companies:
            company_results = self.vector_store.search_by_company_year(
                query, company=company, n_results=n_results // len(companies) + 1
            )
            all_results.extend(company_results)
        
        # Sort by relevance and return top results
        all_results.sort(key=lambda x: x["distance"])
        return all_results[:n_results]
    
    def _temporal_search(self, query: str, years: List[int], n_results: int = 8, **kwargs) -> List[Dict]:
        """Search focused on specific years."""
        all_results = []
        
        for year in years:
            year_results = self.vector_store.search_by_company_year(
                query, year=year, n_results=n_results // len(years) + 1
            )
            all_results.extend(year_results)
        
        # Sort by relevance
        all_results.sort(key=lambda x: x["distance"])
        return all_results[:n_results]
    
    def multi_query_retrieval(self, queries: List[str], strategy: str = "semantic", n_results_per_query: int = 5) -> List[Dict]:
        """Retrieve results for multiple queries and combine them."""
        all_results = []
        seen_chunks = set()
        
        for query in queries:
            query_results = self.retrieve_for_query(query, strategy, n_results=n_results_per_query)
            
            for result in query_results:
                # Use chunk text as identifier to avoid duplicates
                chunk_id = result["metadata"].get("chunk_id", result["text"][:100])
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    result["source_query"] = query
                    all_results.append(result)
        
        # Sort by relevance
        all_results.sort(key=lambda x: x["distance"])
        return all_results
