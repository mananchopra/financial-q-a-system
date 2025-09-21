"""Main orchestrator for the financial Q&A agent system."""
import json
from typing import Dict, List, Optional
import google.generativeai as genai
from rich.console import Console

from .query_classifier import QueryClassifier, QueryDecomposer, QueryType
from .synthesizer import ResultSynthesizer
from ..rag.vector_store import FinancialVectorStore, RetrievalEngine

console = Console()

class FinancialQAAgent:
    """Main agent orchestrator for financial Q&A system."""
    
    def __init__(self, vector_store: FinancialVectorStore, google_api_key: str, model: str = "gemini-1.5-flash"):
        self.vector_store = vector_store
        genai.configure(api_key=google_api_key)
        
        self.query_classifier = QueryClassifier(model)
        self.query_decomposer = QueryDecomposer(model)
        self.retrieval_engine = RetrievalEngine(vector_store)
        self.synthesizer = ResultSynthesizer(model)
    
    def answer_query(self, query: str, verbose: bool = False) -> Dict:
        """Answer a financial query using the agent workflow."""
        
        if verbose:
            console.print(f"[bold blue]Processing query:[/bold blue] {query}")
        
        try:
            query_type, classification_info = self.query_classifier.classify_query(query)
            
            if verbose:
                console.print(f"[blue]Query type:[/blue] {query_type.value}")
                console.print(f"[blue]Classification info:[/blue] {classification_info}")
            
            sub_queries = self.query_decomposer.decompose_query(
                query, query_type, classification_info
            )
            
            if verbose:
                console.print(f"[blue]Sub-queries:[/blue] {sub_queries}")
            
            retrieval_results = {}
            for sub_query in sub_queries:
                results = self._execute_retrieval(sub_query, query_type, classification_info, verbose)
                retrieval_results[sub_query] = results
            
            final_answer = self.synthesizer.synthesize_answer(
                query, sub_queries, retrieval_results, query_type.value
            )
            
            if verbose:
                console.print(f"[green]Final answer generated[/green]")
            
            return final_answer
            
        except Exception as e:
            console.print(f"[red]Error processing query: {e}[/red]")
            return {
                "query": query,
                "answer": f"An error occurred while processing your query: {str(e)}",
                "reasoning": "System error during processing",
                "sub_queries": [],
                "sources": [],
                "confidence": "low"
            }
    
    def _execute_retrieval(self, sub_query: str, query_type: QueryType, classification_info: Dict, verbose: bool = False) -> List[Dict]:
        """Execute retrieval for a single sub-query."""
        if query_type == QueryType.CROSS_COMPANY:
            companies = classification_info.get("companies", ["GOOGL", "MSFT", "NVDA"])
            results = self.retrieval_engine.retrieve_for_query(
                sub_query, 
                strategy="company_focused",
                companies=companies,
                n_results=6
            )
        elif query_type == QueryType.COMPARATIVE_YOY:
            years = classification_info.get("years", [])
            if years:
                results = self.retrieval_engine.retrieve_for_query(
                    sub_query,
                    strategy="temporal", 
                    years=years,
                    n_results=6
                )
            else:
                results = self.retrieval_engine.retrieve_for_query(
                    sub_query,
                    strategy="hybrid",
                    n_results=6
                )
        else:
            results = self.retrieval_engine.retrieve_for_query(
                sub_query,
                strategy="hybrid",
                n_results=6
            )
        
        if verbose and results:
            console.print(f"[yellow]Retrieved {len(results)} results for: {sub_query}[/yellow]")
        
        return results
    
    def batch_answer_queries(self, queries: List[str], verbose: bool = False) -> List[Dict]:
        """Answer multiple queries in batch."""
        results = []
        
        console.print(f"[bold blue]Processing {len(queries)} queries...[/bold blue]")
        
        for i, query in enumerate(queries, 1):
            console.print(f"\n[bold]Query {i}/{len(queries)}:[/bold]")
            result = self.answer_query(query, verbose)
            results.append(result)
        
        return results
    
    def get_system_stats(self) -> Dict:
        """Get system statistics and status."""
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            "vector_store": vector_stats,
            "agent_components": {
                "query_classifier": "active",
                "query_decomposer": "active", 
                "retrieval_engine": "active",
                "synthesizer": "active"
            },
            "supported_query_types": [qtype.value for qtype in QueryType]
        }

class QueryProcessor:
    """Handles query preprocessing and validation."""
    
    @staticmethod
    def preprocess_query(query: str) -> str:
        """Clean and preprocess the input query."""
        query = query.strip()
        
        replacements = {
            "alphabet": "google",
            "googl": "google", 
            "msft": "microsoft",
            "nvda": "nvidia"
        }
        
        query_lower = query.lower()
        for old, new in replacements.items():
            query_lower = query_lower.replace(old, new)
        words = query.split()
        processed_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in ["google", "microsoft", "nvidia"]:
                processed_words.append(word_lower.title())
            else:
                processed_words.append(word)
        
        return " ".join(processed_words)
    
    @staticmethod
    def validate_query(query: str) -> tuple[bool, str]:
        """Validate if query is appropriate for the system."""
        if len(query.strip()) < 5:
            return False, "Query too short"
        
        if len(query) > 500:
            return False, "Query too long"
        
        financial_keywords = [
            "revenue", "income", "profit", "sales", "margin", "earnings",
            "financial", "money", "dollar", "billion", "million", "growth",
            "year", "annual", "quarterly", "fiscal", "operating", "net"
        ]
        
        query_lower = query.lower()
        has_financial_keyword = any(keyword in query_lower for keyword in financial_keywords)
        
        if not has_financial_keyword:
            return False, "Query doesn't appear to be financial-related"
        
        return True, "Valid query"

class ResponseFormatter:
    """Formats agent responses for different output formats."""
    
    @staticmethod
    def format_json(response: Dict, pretty: bool = True) -> str:
        """Format response as JSON."""
        if pretty:
            return json.dumps(response, indent=2, ensure_ascii=False)
        return json.dumps(response, ensure_ascii=False)
    
    @staticmethod
    def format_text(response: Dict) -> str:
        """Format response as readable text."""
        text_parts = []
        
        text_parts.append(f"Query: {response['query']}")
        text_parts.append(f"Answer: {response['answer']}")
        
        if response.get('reasoning'):
            text_parts.append(f"Reasoning: {response['reasoning']}")
        
        if response.get('sub_queries'):
            text_parts.append(f"Sub-queries analyzed: {', '.join(response['sub_queries'])}")
        
        if response.get('sources'):
            text_parts.append("\nSources:")
            for i, source in enumerate(response['sources'], 1):
                text_parts.append(f"  {i}. {source['company']} {source['year']}: {source['excerpt']}")
        
        text_parts.append(f"Confidence: {response.get('confidence', 'unknown')}")
        
        return "\n".join(text_parts)
    
    @staticmethod
    def format_markdown(response: Dict) -> str:
        """Format response as markdown."""
        md_parts = []
        
        md_parts.append(f"## Query\n{response['query']}\n")
        md_parts.append(f"## Answer\n{response['answer']}\n")
        
        if response.get('reasoning'):
            md_parts.append(f"## Reasoning\n{response['reasoning']}\n")
        
        if response.get('sub_queries'):
            md_parts.append("## Sub-queries Analyzed")
            for sq in response['sub_queries']:
                md_parts.append(f"- {sq}")
            md_parts.append("")
        
        if response.get('sources'):
            md_parts.append("## Sources")
            for i, source in enumerate(response['sources'], 1):
                md_parts.append(f"{i}. **{source['company']} {source['year']}**: {source['excerpt']}")
            md_parts.append("")
        
        md_parts.append(f"**Confidence**: {response.get('confidence', 'unknown')}")
        
        return "\n".join(md_parts)
