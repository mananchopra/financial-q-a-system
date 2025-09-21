"""SEC filing downloader for 10-K documents."""
import os
import re
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, TaskID

console = Console()

class SECDownloader:
    """Downloads SEC 10-K filings for specified companies and years."""
    
    def __init__(self, companies: Dict[str, Dict], years: List[int], data_dir: str):
        self.companies = companies
        self.years = years
        self.data_dir = Path(data_dir)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Financial Q&A System research@example.com",
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov"
        })
        
    def download_all_filings(self) -> None:
        """Download all 10-K filings for configured companies and years."""
        console.print("[bold blue]Starting SEC filing downloads...[/bold blue]")
        
        total_filings = len(self.companies) * len(self.years)
        
        with Progress() as progress:
            task = progress.add_task("Downloading filings...", total=total_filings)
            
            for symbol, info in self.companies.items():
                for year in self.years:
                    try:
                        self.download_10k(symbol, info["cik"], year)
                        progress.advance(task)
                        time.sleep(0.5)
                    except Exception as e:
                        console.print(f"[red]Error downloading {symbol} {year}: {e}[/red]")
                        progress.advance(task)
                        
        console.print("[bold green]Download complete![/bold green]")
    
    def download_10k(self, symbol: str, cik: str, year: int) -> Optional[str]:
        """Download a specific 10-K filing."""
        # Create directory structure
        filing_dir = self.data_dir / symbol / str(year)
        filing_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        html_file = filing_dir / "10k.html"
        if html_file.exists():
            console.print(f"[yellow]Skipping {symbol} {year} - already exists[/yellow]")
            return str(html_file)
        
        try:
            # Get filing list for the company
            console.print(f"[blue]Downloading {symbol} 10-K for {year}...[/blue]")
            filing_url = self._find_10k_filing(cik, year)
            
            if not filing_url:
                console.print(f"[red]Could not find 10-K for {symbol} {year}[/red]")
                return None
                
            # Download the filing
            response = self.session.get(filing_url, timeout=30)
            response.raise_for_status()
            
            # Save the filing
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
                
            console.print(f"[green]Downloaded {symbol} {year} ✓[/green]")
            return str(html_file)
            
        except Exception as e:
            console.print(f"[red]Failed to download {symbol} {year}: {e}[/red]")
            return None
    
    def _find_10k_filing(self, cik: str, year: int) -> Optional[str]:
        """Find the 10-K filing URL for a given company and year."""
        # SEC EDGAR search URL
        search_url = f"https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            "action": "getcompany",
            "CIK": cik,
            "type": "10-K",
            "dateb": f"{year}1231",  # End of year
            "count": "10"
        }
        
        try:
            response = self.session.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the 10-K filing for the specific year
            for row in soup.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 4:
                    # Check if this is a 10-K filing
                    filing_type = cells[0].get_text(strip=True)
                    filing_date = cells[3].get_text(strip=True)
                    
                    if filing_type == "10-K" and str(year) in filing_date:
                        # Get the documents link
                        docs_link = cells[1].find('a')
                        if docs_link:
                            docs_url = "https://www.sec.gov" + docs_link['href']
                            return self._get_html_filing_url(docs_url)
            
            return None
            
        except Exception as e:
            console.print(f"[red]Error searching for filing: {e}[/red]")
            return None
    
    def _get_html_filing_url(self, docs_url: str) -> Optional[str]:
        """Get the actual HTML filing URL from the documents page."""
        try:
            response = self.session.get(docs_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for the main 10-K document (usually the first .htm file)
            for row in soup.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 3:
                    # Check if this is the main document
                    doc_name = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                    if doc_name and (".htm" in doc_name.lower() or ".html" in doc_name.lower()):
                        # Skip if it's clearly an exhibit or attachment
                        if not any(word in doc_name.lower() for word in ["ex-", "exhibit", "attachment"]):
                            link = cells[2].find('a')
                            if link:
                                return "https://www.sec.gov" + link['href']
            
            return None
            
        except Exception as e:
            console.print(f"[red]Error getting HTML filing URL: {e}[/red]")
            return None

# Simple alternative: Manual filing URLs (fallback approach)
MANUAL_FILING_URLS = {
        "GOOGL": {
            2022: "https://www.sec.gov/Archives/edgar/data/1652044/000165204423000016/goog-20221231.htm",
            2023: "https://www.sec.gov/Archives/edgar/data/1652044/000165204424000022/goog-20231231.htm", 
            2024: "https://www.sec.gov/Archives/edgar/data/1652044/000165204425000014/goog-20241231.htm"
        },
    "MSFT": {
        2022: "https://www.sec.gov/Archives/edgar/data/789019/000156459022026876/msft-10k_20220630.htm",
        2023: "https://www.sec.gov/Archives/edgar/data/789019/000095017023035122/msft-20230630.htm",
        2024: "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm"
    },
    "NVDA": {
        2022: "https://www.sec.gov/Archives/edgar/data/1045810/000104581022000036/nvda-20220130.htm",
        2023: "https://www.sec.gov/Archives/edgar/data/1045810/000104581023000017/nvda-20230129.htm",
        2024: "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm"
    }
}

class SimpleSECDownloader:
    """Simplified downloader using known filing URLs."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Financial Q&A System research@example.com"
        })
    
    def download_all_filings(self) -> None:
        """Download all filings using predefined URLs."""
        console.print("[bold blue]Starting simplified SEC filing downloads...[/bold blue]")
        
        total_filings = sum(len(years) for years in MANUAL_FILING_URLS.values())
        
        with Progress() as progress:
            task = progress.add_task("Downloading filings...", total=total_filings)
            
            for symbol, year_urls in MANUAL_FILING_URLS.items():
                for year, url in year_urls.items():
                    try:
                        self._download_filing(symbol, year, url)
                        progress.advance(task)
                        time.sleep(0.5)
                    except Exception as e:
                        console.print(f"[red]Error downloading {symbol} {year}: {e}[/red]")
                        progress.advance(task)
                        
        console.print("[bold green]Download complete![/bold green]")
    
    def _download_filing(self, symbol: str, year: int, url: str) -> None:
        """Download a specific filing from URL."""
        # Create directory
        filing_dir = self.data_dir / symbol / str(year)
        filing_dir.mkdir(parents=True, exist_ok=True)
        
        html_file = filing_dir / "10k.html"
        
        # Skip if already exists
        if html_file.exists():
            console.print(f"[yellow]Skipping {symbol} {year} - already exists[/yellow]")
            return
        
        # Download filing
        console.print(f"[blue]Downloading {symbol} 10-K for {year}...[/blue]")
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        
        # Save filing
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
            
        console.print(f"[green]Downloaded {symbol} {year} ✓[/green]")
