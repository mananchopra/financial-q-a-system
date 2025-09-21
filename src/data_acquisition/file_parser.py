"""Text extraction and parsing for SEC filings."""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup, Tag
from rich.console import Console

console = Console()

class SECFilingParser:
    """Extracts and structures text from SEC 10-K HTML filings."""
    
    def __init__(self):
        self.important_sections = {
            "item_1": ["item 1", "business"],
            "item_1a": ["item 1a", "risk factors"], 
            "item_2": ["item 2", "properties"],
            "item_3": ["item 3", "legal proceedings"],
            "item_7": ["item 7", "management's discussion", "md&a"],
            "item_8": ["item 8", "financial statements"],
            "item_9": ["item 9", "controls and procedures"]
        }
    
    def parse_filing(self, file_path: str, company: str, year: int) -> Dict:
        """Parse a 10-K filing and extract structured text."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            self._clean_html(soup)
            filing_info = {
                "company": company,
                "year": year,
                "file_path": file_path,
                "sections": {},
                "full_text": ""
            }
            
            full_text = soup.get_text()
            filing_info["full_text"] = self._clean_text(full_text)
            
            sections = self._extract_sections(soup)
            filing_info["sections"] = sections
            
            console.print(f"[green]Parsed {company} {year} filing[/green]")
            return filing_info
            
        except Exception as e:
            console.print(f"[red]Error parsing {file_path}: {e}[/red]")
            return {"company": company, "year": year, "error": str(e)}
    
    def _clean_html(self, soup: BeautifulSoup) -> None:
        """Remove unnecessary HTML elements."""
        for tag in soup(["script", "style", "meta", "link", "noscript"]):
            tag.decompose()
        
        for table in soup.find_all("table"):
            if self._is_formatting_table(table):
                table.decompose()
    
    def _is_formatting_table(self, table: Tag) -> bool:
        """Check if a table is for formatting rather than financial data."""
        text = table.get_text().lower()
        if len(text) < 100 or "page" in text or "table of contents" in text:
            return True
        return False
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\f\r]+', ' ', text)
        text = re.sub(r'\.{3,}', '...', text)
        return text.strip()
    
    def _extract_sections(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract important sections from the filing."""
        sections = {}
        full_text = soup.get_text().lower()
        
        for section_key, keywords in self.important_sections.items():
            section_text = self._find_section_text(full_text, keywords)
            if section_text:
                sections[section_key] = self._clean_text(section_text)
        
        return sections
    
    def _find_section_text(self, full_text: str, keywords: List[str]) -> Optional[str]:
        """Find and extract text for a specific section."""
        patterns = []
        for keyword in keywords:
            if keyword.startswith("item"):
                item_num = keyword.split()[1] if len(keyword.split()) > 1 else ""
                patterns.extend([
                    rf"item\s+{re.escape(item_num)}[\.\s\-:]*",
                    rf"item\s*{re.escape(item_num)}[\.\s\-:]*"
                ])
            else:
                patterns.append(rf"{re.escape(keyword)}")
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
            if matches:
                start_pos = matches[0].start()
                
                next_item_pattern = r"item\s+\d+[a-z]?[\.\s\-:]"
                next_matches = list(re.finditer(next_item_pattern, full_text[start_pos + 100:], re.IGNORECASE))
                
                if next_matches:
                    end_pos = start_pos + 100 + next_matches[0].start()
                    return full_text[start_pos:end_pos]
                else:
                    remaining = full_text[start_pos:]
                    return remaining[:50000] if len(remaining) > 50000 else remaining
        
        return None

class DocumentProcessor:
    """Process multiple filings and prepare them for RAG pipeline."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.parser = SECFilingParser()
    
    def process_all_filings(self) -> List[Dict]:
        """Process all downloaded filings."""
        documents = []
        
        console.print("[bold blue]Processing SEC filings...[/bold blue]")
        
        for company_dir in self.data_dir.iterdir():
            if company_dir.is_dir():
                company = company_dir.name
                
                for year_dir in company_dir.iterdir():
                    if year_dir.is_dir():
                        year = int(year_dir.name)
                        filing_path = year_dir / "10k.html"
                        
                        if filing_path.exists():
                            doc = self.parser.parse_filing(str(filing_path), company, year)
                            if "error" not in doc:
                                documents.append(doc)
                        else:
                            console.print(f"[yellow]Missing filing: {filing_path}[/yellow]")
        
        console.print(f"[green]Processed {len(documents)} filings[/green]")
        return documents
    
    def get_financial_metrics_text(self, documents: List[Dict]) -> List[Dict]:
        """Extract text chunks focused on financial metrics."""
        financial_chunks = []
        
        for doc in documents:
            company = doc["company"]
            year = doc["year"]
            
            for section_key in ["item_7", "item_8"]:
                if section_key in doc["sections"]:
                    section_text = doc["sections"][section_key]
                    
                    chunks = self._extract_financial_chunks(section_text, company, year)
                    financial_chunks.extend(chunks)
        
        return financial_chunks
    
    def _extract_financial_chunks(self, text: str, company: str, year: int) -> List[Dict]:
        """Extract chunks containing financial metrics."""
        chunks = []
        
        financial_keywords = [
            "revenue", "income", "earnings", "profit", "margin", 
            "sales", "operating", "net income", "gross", "ebitda",
            "cash flow", "assets", "liabilities", "equity",
            "billion", "million", "percent", "%", "$"
        ]
        
        paragraphs = text.split('\n')
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 100:
                lower_para = paragraph.lower()
                financial_score = sum(1 for keyword in financial_keywords if keyword in lower_para)
                
                if financial_score >= 2:
                    chunks.append({
                        "text": paragraph.strip(),
                        "company": company,
                        "year": year,
                        "section": "financial_metrics",
                        "chunk_id": f"{company}_{year}_financial_{i}",
                        "financial_score": financial_score
                    })
        
        return chunks
