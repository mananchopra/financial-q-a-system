"""Main CLI interface for the Financial Q&A System."""
import os
import sys
import click
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from config.settings import (
    GOOGLE_API_KEY, CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME,
    COMPANIES, YEARS, DATA_DIR
)
from src.data_acquisition.sec_downloader import SimpleSECDownloader
from src.data_acquisition.file_parser import DocumentProcessor
from src.rag.chunking import FinancialTextChunker
from src.rag.vector_store import FinancialVectorStore
from src.agents.orchestrator import FinancialQAAgent, QueryProcessor, ResponseFormatter

console = Console()

@click.group()
def cli():
    """Financial Q&A System - RAG with Agent Capabilities for SEC 10-K filings."""
    pass

@cli.command()
def setup():
    """Download SEC filings and set up the system."""
    
    console.print("[bold blue]Setting up Financial Q&A System...[/bold blue]")
    
    if not GOOGLE_API_KEY:
        console.print("[red]Error: GOOGLE_API_KEY not found in environment.[/red]")
        console.print("Please set your Google API key in a .env file or environment variable.")
        return
    
    try:
        console.print("\n[bold]Step 1: Downloading SEC filings...[/bold]")
        downloader = SimpleSECDownloader(DATA_DIR)
        downloader.download_all_filings()
        
        console.print("\n[bold]Step 2: Processing filings...[/bold]")
        processor = DocumentProcessor(DATA_DIR)
        documents = processor.process_all_filings()
        
        if not documents:
            console.print("[red]No documents were processed successfully.[/red]")
            return
        
        console.print("\n[bold]Step 3: Chunking documents...[/bold]")
        chunker = FinancialTextChunker()
        chunks = chunker.chunk_documents(documents)
        
        console.print("\n[bold]Step 4: Setting up vector store...[/bold]")
        vector_store = FinancialVectorStore(
            CHROMA_PERSIST_DIRECTORY, 
            COLLECTION_NAME, 
            GOOGLE_API_KEY
        )
        
        vector_store.clear_collection()
        vector_store.add_documents(chunks)
        
        stats = vector_store.get_collection_stats()
        
        table = Table(title="Setup Complete - System Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Chunks", str(stats["total_chunks"]))
        table.add_row("Companies", ", ".join(stats["companies"]))
        table.add_row("Years", ", ".join(stats["years"]))
        table.add_row("Sections", str(len(stats["sections"])))
        
        console.print(table)
        console.print("\n[bold green]System setup complete! You can now run queries.[/bold green]")
        
    except Exception as e:
        console.print(f"[red]Setup failed: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())

@cli.command()
@click.argument('query', required=False)
@click.option('--verbose', '-v', is_flag=True, help='Verbose output showing agent workflow')
@click.option('--format', '-f', default='text', type=click.Choice(['json', 'text', 'markdown']), help='Output format')
@click.option('--pretty', '-p', is_flag=True, help='Pretty print JSON output')
def query(query, verbose, format, pretty):
    """Ask a financial question about Google, Microsoft, or NVIDIA."""
    
    if not GOOGLE_API_KEY:
        console.print("[red]Error: GOOGLE_API_KEY not found in environment.[/red]")
        return
    if not query:
        console.print("[bold blue]Financial Q&A System - Interactive Mode[/bold blue]")
        console.print("Ask questions about Google, Microsoft, or NVIDIA's financial data from their 10-K filings.")
        console.print("Type 'quit' to exit.\n")
        
        try:
            vector_store = FinancialVectorStore(
                CHROMA_PERSIST_DIRECTORY, 
                COLLECTION_NAME, 
                GOOGLE_API_KEY
            )
            agent = FinancialQAAgent(vector_store, GOOGLE_API_KEY)
        except Exception as e:
            console.print(f"[red]Failed to initialize system: {e}[/red]")
            console.print("Run 'python main.py setup' first to set up the system.")
            return
        
        while True:
            try:
                user_query = click.prompt("\n🤖 Your question", type=str)
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    console.print("Goodbye!")
                    break
                
                response = process_single_query(agent, user_query, verbose)
                if format == 'json':
                    console.print(ResponseFormatter.format_json(response, pretty))
                elif format == 'markdown':
                    console.print(ResponseFormatter.format_markdown(response))
                else:
                    console.print(ResponseFormatter.format_text(response))
                    
            except KeyboardInterrupt:
                console.print("\nGoodbye!")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    else:
        try:
            vector_store = FinancialVectorStore(
                CHROMA_PERSIST_DIRECTORY, 
                COLLECTION_NAME, 
                GOOGLE_API_KEY
            )
            agent = FinancialQAAgent(vector_store, GOOGLE_API_KEY)
            
            response = process_single_query(agent, query, verbose)
            if format == 'json':
                console.print(ResponseFormatter.format_json(response, pretty))
            elif format == 'markdown':
                console.print(ResponseFormatter.format_markdown(response))
            else:
                console.print(ResponseFormatter.format_text(response))
                
        except Exception as e:
            console.print(f"[red]Failed to process query: {e}[/red]")

@cli.command()
def test():
    """Run test queries to validate system functionality."""
    
    test_queries = [
        "What was NVIDIA's total revenue in fiscal year 2024?",
        "What percentage of Google's 2023 revenue came from advertising?",
        "How much did Microsoft's cloud revenue grow from 2022 to 2023?",
        "Which of the three companies had the highest gross margin in 2023?",
        "Compare the R&D spending as a percentage of revenue across all three companies in 2023"
    ]
    
    console.print("[bold blue]Running test queries...[/bold blue]")
    
    try:
        vector_store = FinancialVectorStore(
            CHROMA_PERSIST_DIRECTORY, 
            COLLECTION_NAME, 
            GOOGLE_API_KEY
        )
        agent = FinancialQAAgent(vector_store, GOOGLE_API_KEY)
        
        results = agent.batch_answer_queries(test_queries, verbose=True)
        
        # Display results
        for i, (query, result) in enumerate(zip(test_queries, results), 1):
            console.print(f"\n[bold]Test Query {i}:[/bold]")
            console.print(Panel(ResponseFormatter.format_text(result), title=f"Query {i}"))
            
    except Exception as e:
        console.print(f"[red]Test failed: {e}[/red]")

@cli.command()
def stats():
    """Show system statistics."""
    
    try:
        vector_store = FinancialVectorStore(
            CHROMA_PERSIST_DIRECTORY, 
            COLLECTION_NAME, 
            GOOGLE_API_KEY
        )
        agent = FinancialQAAgent(vector_store, GOOGLE_API_KEY)
        
        stats = agent.get_system_stats()
        
        console.print(json.dumps(stats, indent=2))
        
    except Exception as e:
        console.print(f"[red]Failed to get stats: {e}[/red]")

def process_single_query(agent: FinancialQAAgent, query: str, verbose: bool = False) -> dict:
    """Process a single query with validation."""
    
    # Preprocess query
    processed_query = QueryProcessor.preprocess_query(query)
    
    # Validate query
    is_valid, message = QueryProcessor.validate_query(processed_query)
    if not is_valid:
        return {
            "query": query,
            "answer": f"Invalid query: {message}",
            "reasoning": "Query validation failed",
            "sub_queries": [],
            "sources": [],
            "confidence": "low"
        }
    
    # Process with agent
    return agent.answer_query(processed_query, verbose)

if __name__ == "__main__":
    cli()
