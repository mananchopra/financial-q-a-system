"""Text chunking strategies for financial documents."""
import re
import google.generativeai as genai
from typing import List, Dict, Optional
from rich.console import Console

console = Console()

class FinancialTextChunker:
    """Intelligent chunking for financial documents."""
    
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100, model: str = "gemini-1.5-flash"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        
        self.section_patterns = [
            r"item\s+\d+[a-z]?[\.\s\-:]",
            r"part\s+[ivx]+",
            r"table\s+of\s+contents",
            r"financial\s+statements",
            r"management['\s]s\s+discussion",
            r"risk\s+factors"
        ]
        
        self.metrics_patterns = [
            r"\$[\d,]+\.?\d*\s*(million|billion|thousand)?",
            r"\d+\.?\d*\s*%",
            r"revenue|income|earnings|margin|profit|sales",
            r"fiscal\s+year\s+\d{4}",
            r"quarter|quarterly|annual|yearly"
        ]
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Chunk all documents into smaller pieces with metadata."""
        all_chunks = []
        
        console.print("[bold blue]Chunking documents...[/bold blue]")
        
        for doc in documents:
            if "error" in doc:
                continue
                
            company = doc["company"]
            year = doc["year"]
            
            full_text_chunks = self._chunk_text(
                doc["full_text"], 
                company, 
                year, 
                "full_document"
            )
            all_chunks.extend(full_text_chunks)
            
            # Chunk important sections separately for better retrieval
            for section_name, section_text in doc.get("sections", {}).items():
                if section_text and len(section_text) > 200:
                    section_chunks = self._chunk_text(
                        section_text,
                        company,
                        year, 
                        section_name
                    )
                    all_chunks.extend(section_chunks)
        
        console.print(f"[green]Created {len(all_chunks)} chunks[/green]")
        return all_chunks
    
    def _chunk_text(self, text: str, company: str, year: int, section: str) -> List[Dict]:
        """Chunk a single text into overlapping pieces."""
        # Clean text first
        text = self._preprocess_text(text)
        
        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = len(text) // 4
        
        if estimated_tokens <= self.chunk_size:
            # Text is small enough, return as single chunk
            return [self._create_chunk(text, company, year, section, 0)]
        
        chunks = []
        chunk_num = 0
        
        # Split by estimated character count
        chars_per_chunk = self.chunk_size * 4
        chars_overlap = self.chunk_overlap * 4
        
        start_idx = 0
        while start_idx < len(text):
            end_idx = min(start_idx + chars_per_chunk, len(text))
            chunk_text = text[start_idx:end_idx]
            
            # Try to end chunk at sentence boundary if possible
            if end_idx < len(text):  # Not the last chunk
                chunk_text = self._adjust_chunk_boundary(chunk_text)
            
            # Create chunk with metadata
            chunk = self._create_chunk(chunk_text, company, year, section, chunk_num)
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            if end_idx >= len(text):
                break
            start_idx = end_idx - chars_overlap
            chunk_num += 1
        
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for chunking."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'table\s+of\s+contents.*?(?=item\s+1)', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Normalize financial amounts for better matching
        text = re.sub(r'\$\s+(\d)', r'$\1', text)  # Remove space after $
        
        return text.strip()
    
    def _adjust_chunk_boundary(self, chunk_text: str) -> str:
        """Adjust chunk boundary to end at sentence boundary if possible."""
        # Try to end at sentence boundary
        sentences = re.split(r'[.!?]+\s+', chunk_text)
        
        if len(sentences) > 1:
            # Remove last incomplete sentence if chunk is long enough
            complete_text = '. '.join(sentences[:-1]) + '.'
            
            # Only use adjusted boundary if it's not too short
            if len(complete_text) > len(chunk_text) * 0.7:
                return complete_text
        
        return chunk_text
    
    def _create_chunk(self, text: str, company: str, year: int, section: str, chunk_num: int) -> Dict:
        """Create a chunk dictionary with metadata."""
        # Calculate metrics for this chunk
        financial_score = self._calculate_financial_score(text)
        
        return {
            "text": text,
            "company": company,
            "year": year,
            "section": section,
            "chunk_id": f"{company}_{year}_{section}_{chunk_num}",
            "chunk_number": chunk_num,
            "token_count": len(text) // 4,  # Rough approximation
            "financial_score": financial_score,
            "metadata": {
                "company": company,
                "year": year,
                "section": section,
                "chunk_number": chunk_num,
                "has_financial_data": financial_score > 2
            }
        }
    
    def _calculate_financial_score(self, text: str) -> int:
        """Calculate how much financial information this chunk contains."""
        score = 0
        lower_text = text.lower()
        
        # Check for financial metrics
        for pattern in self.metrics_patterns:
            matches = re.findall(pattern, lower_text)
            score += len(matches)
        
        # Bonus for specific financial keywords
        financial_keywords = [
            "revenue", "income", "profit", "margin", "earnings",
            "sales", "operating", "net income", "gross margin",
            "ebitda", "cash flow", "total assets", "shareholders equity"
        ]
        
        for keyword in financial_keywords:
            if keyword in lower_text:
                score += 1
        
        return score

class SmartChunker:
    """Advanced chunking that preserves financial context."""
    
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_with_context_preservation(self, documents: List[Dict]) -> List[Dict]:
        """Chunk documents while preserving financial context."""
        all_chunks = []
        
        for doc in documents:
            if "error" in doc:
                continue
            
            # Process different sections with different strategies
            company = doc["company"]
            year = doc["year"]
            
            # Financial statements need special handling
            if "item_8" in doc.get("sections", {}):
                financial_chunks = self._chunk_financial_statements(
                    doc["sections"]["item_8"], company, year
                )
                all_chunks.extend(financial_chunks)
            
            # MD&A section
            if "item_7" in doc.get("sections", {}):
                mda_chunks = self._chunk_mda_section(
                    doc["sections"]["item_7"], company, year
                )
                all_chunks.extend(mda_chunks)
            
            # Risk factors
            if "item_1a" in doc.get("sections", {}):
                risk_chunks = self._chunk_risk_factors(
                    doc["sections"]["item_1a"], company, year
                )
                all_chunks.extend(risk_chunks)
        
        return all_chunks
    
    def _chunk_financial_statements(self, text: str, company: str, year: int) -> List[Dict]:
        """Special chunking for financial statements."""
        chunks = []
        
        # Look for financial statement subsections
        subsections = self._identify_financial_subsections(text)
        
        chunk_num = 0
        for subsection_name, subsection_text in subsections.items():
            if len(subsection_text) > 100:
                # Chunk each subsection
                subsection_chunks = self._basic_chunk(
                    subsection_text, company, year, f"financial_{subsection_name}", chunk_num
                )
                chunks.extend(subsection_chunks)
                chunk_num += len(subsection_chunks)
        
        return chunks
    
    def _identify_financial_subsections(self, text: str) -> Dict[str, str]:
        """Identify different parts of financial statements."""
        subsections = {}
        
        # Common financial statement sections
        section_patterns = {
            "income_statement": [
                "consolidated statements of income",
                "consolidated income statements", 
                "statements of operations"
            ],
            "balance_sheet": [
                "consolidated balance sheets",
                "consolidated statements of financial position"
            ],
            "cash_flow": [
                "consolidated statements of cash flows",
                "cash flow statements"
            ],
            "equity": [
                "consolidated statements of equity",
                "statements of shareholders"
            ]
        }
        
        # Default to full text if no sections found
        subsections["all"] = text
        
        return subsections
    
    def _basic_chunk(self, text: str, company: str, year: int, section: str, start_chunk_num: int) -> List[Dict]:
        """Basic chunking with metadata."""
        chunker = FinancialTextChunker(self.chunk_size, self.chunk_overlap)
        chunks = chunker._chunk_text(text, company, year, section)
        
        # Adjust chunk numbers
        for i, chunk in enumerate(chunks):
            chunk["chunk_number"] = start_chunk_num + i
            chunk["chunk_id"] = f"{company}_{year}_{section}_{start_chunk_num + i}"
        
        return chunks
    
    def _chunk_mda_section(self, text: str, company: str, year: int) -> List[Dict]:
        """Chunk MD&A section preserving business context."""
        return self._basic_chunk(text, company, year, "mda", 0)
    
    def _chunk_risk_factors(self, text: str, company: str, year: int) -> List[Dict]:
        """Chunk risk factors section."""
        return self._basic_chunk(text, company, year, "risk_factors", 0)
