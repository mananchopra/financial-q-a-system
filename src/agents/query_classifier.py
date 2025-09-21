"""Query classification and decomposition for financial Q&A."""
import re
from typing import Dict, List, Optional, Tuple
from enum import Enum
import google.generativeai as genai
from rich.console import Console

console = Console()

class QueryType(Enum):
    SIMPLE_DIRECT = "simple_direct"
    COMPARATIVE_YOY = "comparative_yoy" 
    CROSS_COMPANY = "cross_company"
    COMPLEX_MULTI_ASPECT = "complex_multi_aspect"
    SEGMENT_ANALYSIS = "segment_analysis"

class QueryClassifier:
    """Classifies financial queries and determines processing strategy."""
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        
        self.patterns = {
            QueryType.SIMPLE_DIRECT: [
                r"what (was|is) .+ (revenue|income|profit|margin)",
                r"(revenue|income|sales|profit) .+ (in|for) \d{4}",
                r"total .+ \d{4}"
            ],
            QueryType.COMPARATIVE_YOY: [
                r"(grow|growth|increase|decrease|change) .+ from \d{4} to \d{4}",
                r"compare .+ \d{4} (and|to|vs) \d{4}",
                r"(year over year|yoy|annually)"
            ],
            QueryType.CROSS_COMPANY: [
                r"which company .+ (highest|lowest|best|worst)",
                r"compare .+ (across|between) .+ (companies|google|microsoft|nvidia)",
                r"(google|microsoft|nvidia) .+ (vs|versus|compared to)"
            ],
            QueryType.SEGMENT_ANALYSIS: [
                r"percentage of .+ revenue",
                r"what portion .+ came from",
                r"breakdown .+ by segment"
            ]
        }
        
        self.company_aliases = {
            "google": ["google", "googl", "alphabet"],
            "microsoft": ["microsoft", "msft"],
            "nvidia": ["nvidia", "nvda"]
        }
    
    def classify_query(self, query: str) -> Tuple[QueryType, Dict]:
        """Classify a query and extract relevant information."""
        query_lower = query.lower()
        
        companies = self._extract_companies(query_lower)
        years = self._extract_years(query_lower)
        metrics = self._extract_metrics(query_lower)
        query_type = self._classify_by_patterns(query_lower)
        
        if query_type is None:
            query_type = self._classify_with_llm(query)
        
        classification_info = {
            "type": query_type,
            "companies": companies,
            "years": years,
            "metrics": metrics,
            "complexity_score": self._calculate_complexity(query_lower, companies, years, metrics)
        }
        
        return query_type, classification_info
    
    def _extract_companies(self, query: str) -> List[str]:
        """Extract company names from query."""
        companies = []
        
        for canonical_name, aliases in self.company_aliases.items():
            for alias in aliases:
                if alias in query:
                    if canonical_name not in companies:
                        companies.append(canonical_name.upper())
        
        return companies
    
    def _extract_years(self, query: str) -> List[int]:
        """Extract years from query."""
        years = []
        year_matches = re.findall(r'\b(20\d{2})\b', query)
        
        for year_str in year_matches:
            year = int(year_str)
            if 2020 <= year <= 2025:
                years.append(year)
        
        return sorted(list(set(years)))
    
    def _extract_metrics(self, query: str) -> List[str]:
        """Extract financial metrics mentioned in query."""
        metrics = []
        
        metric_keywords = [
            "revenue", "sales", "income", "earnings", "profit", "margin",
            "operating margin", "gross margin", "net income", "ebitda",
            "cash flow", "assets", "liabilities", "equity", "expenses",
            "r&d", "research and development", "capex", "operating expenses"
        ]
        
        for metric in metric_keywords:
            if metric in query:
                metrics.append(metric)
        
        return metrics
    
    def _classify_by_patterns(self, query: str) -> Optional[QueryType]:
        """Classify query using regex patterns."""
        for query_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return query_type
        
        return None
    
    def _classify_with_llm(self, query: str) -> QueryType:
        """Use LLM for complex query classification."""
        classification_prompt = f"""
        Classify this financial query into one of these categories:
        
        1. SIMPLE_DIRECT: Asking for a single metric for one company/year
        2. COMPARATIVE_YOY: Comparing metrics across different years
        3. CROSS_COMPANY: Comparing metrics across different companies
        4. COMPLEX_MULTI_ASPECT: Requires multiple calculations/comparisons
        5. SEGMENT_ANALYSIS: Asking about business segment breakdowns
        
        Query: "{query}"
        
        Respond with just the category name.
        """
        
        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                classification_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=50
                )
            )
            
            classification = response.text.strip().upper()
            
            # Map response to enum
            for query_type in QueryType:
                if query_type.value.upper() == classification:
                    return query_type
            
            # Default fallback
            return QueryType.COMPLEX_MULTI_ASPECT
            
        except Exception as e:
            console.print(f"[red]Error in LLM classification: {e}[/red]")
            return QueryType.COMPLEX_MULTI_ASPECT
    
    def _calculate_complexity(self, query: str, companies: List[str], years: List[int], metrics: List[str]) -> int:
        """Calculate query complexity score."""
        score = 1
        
        if len(companies) > 1:
            score += len(companies)
        
        if len(years) > 1:
            score += len(years)
        
        if len(metrics) > 1:
            score += len(metrics)
        
        complex_keywords = ["compare", "growth", "change", "ratio", "percentage", "breakdown"]
        for keyword in complex_keywords:
            if keyword in query:
                score += 1
        
        return score

class QueryDecomposer:
    """Decomposes complex queries into simpler sub-queries."""
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
    
    def decompose_query(self, query: str, query_type: QueryType, classification_info: Dict) -> List[str]:
        """Decompose a complex query into sub-queries."""
        if query_type == QueryType.SIMPLE_DIRECT:
            return [query]  # No decomposition needed
        
        elif query_type == QueryType.COMPARATIVE_YOY:
            return self._decompose_yoy_query(query, classification_info)
        
        elif query_type == QueryType.CROSS_COMPANY:
            return self._decompose_cross_company_query(query, classification_info)
        
        elif query_type == QueryType.COMPLEX_MULTI_ASPECT:
            return self._decompose_complex_query(query, classification_info)
        
        elif query_type == QueryType.SEGMENT_ANALYSIS:
            return self._decompose_segment_query(query, classification_info)
        
        else:
            return [query]
    
    def _decompose_yoy_query(self, query: str, info: Dict) -> List[str]:
        """Decompose year-over-year comparison queries."""
        sub_queries = []
        companies = info.get("companies", [])
        years = info.get("years", [])
        metrics = info.get("metrics", [])
        
        if not companies:
            companies = ["GOOGL", "MSFT", "NVDA"]
        
        if len(years) >= 2:
            for company in companies:
                for metric in metrics or ["revenue"]:
                    for year in years:
                        sub_queries.append(f"{company} {metric} {year}")
        
        return sub_queries or [query]
    
    def _decompose_cross_company_query(self, query: str, info: Dict) -> List[str]:
        """Decompose cross-company comparison queries."""
        sub_queries = []
        companies = info.get("companies", ["GOOGL", "MSFT", "NVDA"])
        years = info.get("years", [])
        metrics = info.get("metrics", [])
        
        target_year = years[0] if years else 2023
        
        for company in companies:
            for metric in metrics or ["operating margin"]:
                sub_queries.append(f"{company} {metric} {target_year}")
        
        return sub_queries
    
    def _decompose_complex_query(self, query: str, info: Dict) -> List[str]:
        """Decompose complex multi-aspect queries using LLM."""
        decomposition_prompt = f"""
        Break down this complex financial query into 2-4 simpler sub-queries that can be answered independently.
        Each sub-query should ask for a specific metric for a specific company and year.
        
        Original query: "{query}"
        Companies mentioned: {info.get('companies', [])}
        Years mentioned: {info.get('years', [])}
        Metrics mentioned: {info.get('metrics', [])}
        
        Format each sub-query as: "[COMPANY] [METRIC] [YEAR]"
        
        Sub-queries:
        """
        
        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                decomposition_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=500
                )
            )
            
            sub_queries_text = response.text.strip()
            
            sub_queries = []
            for line in sub_queries_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('Sub-queries'):
                    line = re.sub(r'^\d+\.\s*', '', line)
                    line = re.sub(r'^-\s*', '', line)
                    sub_queries.append(line.strip())
            
            return sub_queries if sub_queries else [query]
            
        except Exception as e:
            console.print(f"[red]Error in query decomposition: {e}[/red]")
            return [query]
    
    def _decompose_segment_query(self, query: str, info: Dict) -> List[str]:
        """Decompose segment analysis queries."""
        sub_queries = []
        companies = info.get("companies", [])
        years = info.get("years", [])
        
        if not companies:
            if "google" in query.lower():
                companies = ["GOOGL"]
            elif "microsoft" in query.lower():
                companies = ["MSFT"]
            elif "nvidia" in query.lower():
                companies = ["NVDA"]
            else:
                companies = ["GOOGL", "MSFT", "NVDA"]
        
        target_year = years[0] if years else 2023
        
        for company in companies:
            sub_queries.append(f"{company} total revenue {target_year}")
            sub_queries.append(f"{company} segment revenue breakdown {target_year}")
        
        return sub_queries
