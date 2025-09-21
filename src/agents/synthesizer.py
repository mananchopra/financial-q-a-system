"""Result synthesis agent for combining multiple retrieval results."""
import json
import re
from typing import Dict, List, Optional, Any
import google.generativeai as genai
from rich.console import Console

console = Console()

class ResultSynthesizer:
    """Synthesizes multiple retrieval results into coherent answers."""
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
    
    def synthesize_answer(self, query: str, sub_queries: List[str], retrieval_results: Dict[str, List[Dict]], query_type: str) -> Dict:
        """Synthesize final answer from retrieval results."""
        
        context = self._prepare_context(retrieval_results)
        if query_type == "simple_direct":
            answer_data = self._synthesize_simple_answer(query, context)
        elif query_type == "comparative_yoy":
            answer_data = self._synthesize_comparative_answer(query, context, sub_queries)
        elif query_type == "cross_company":
            answer_data = self._synthesize_cross_company_answer(query, context, sub_queries)
        else:
            answer_data = self._synthesize_complex_answer(query, context, sub_queries)
        
        response = {
            "query": query,
            "answer": answer_data.get("answer", "I couldn't find sufficient information to answer this query."),
            "reasoning": answer_data.get("reasoning", "Unable to process the available data."),
            "sub_queries": sub_queries,
            "sources": self._extract_sources(retrieval_results),
            "confidence": answer_data.get("confidence", "low")
        }
        
        return response
    
    def _prepare_context(self, retrieval_results: Dict[str, List[Dict]]) -> str:
        """Prepare context string from retrieval results."""
        context_parts = []
        
        for sub_query, results in retrieval_results.items():
            if results:
                context_parts.append(f"Results for '{sub_query}':")
                for i, result in enumerate(results[:3]):  # Top 3 results per sub-query
                    company = result.get("company", "Unknown")
                    year = result.get("year", "Unknown")
                    text = result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"]
                    context_parts.append(f"  [{i+1}] {company} {year}: {text}")
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _synthesize_simple_answer(self, query: str, context: str) -> Dict:
        """Synthesize answer for simple direct queries."""
        synthesis_prompt = f"""
        Based on the following context from SEC filings, answer this financial query directly and precisely.
        
        Query: {query}
        
        Context:
        {context}
        
        Provide a direct answer with specific numbers and sources. If you find the exact information, state it clearly. If not, explain what information is available.
        
        Format your response as:
        ANSWER: [Direct answer with specific numbers]
        REASONING: [Brief explanation of how you found this information]
        CONFIDENCE: [high/medium/low]
        """
        
        return self._get_llm_response(synthesis_prompt)
    
    def _synthesize_comparative_answer(self, query: str, context: str, sub_queries: List[str]) -> Dict:
        """Synthesize answer for year-over-year comparisons."""
        synthesis_prompt = f"""
        Based on the context from SEC filings, answer this comparative financial query.
        
        Query: {query}
        Sub-queries analyzed: {', '.join(sub_queries)}
        
        Context:
        {context}
        
        Calculate and compare the metrics across the specified time periods. Show:
        1. The specific values for each year
        2. The change (absolute and percentage if applicable)
        3. Any relevant context about the change
        
        Format your response as:
        ANSWER: [Comparison with specific numbers and growth/decline percentages]
        REASONING: [Explanation of the calculation and data sources]
        CONFIDENCE: [high/medium/low]
        """
        
        return self._get_llm_response(synthesis_prompt)
    
    def _synthesize_cross_company_answer(self, query: str, context: str, sub_queries: List[str]) -> Dict:
        """Synthesize answer for cross-company comparisons."""
        synthesis_prompt = f"""
        Based on the context from SEC filings, answer this cross-company comparison query.
        
        Query: {query}
        Companies being compared through sub-queries: {', '.join(sub_queries)}
        
        Context:
        {context}
        
        Compare the metrics across companies and determine:
        1. The specific value for each company
        2. Which company ranks highest/lowest
        3. Any notable differences or context
        
        Format your response as:
        ANSWER: [Clear ranking with specific numbers for each company]
        REASONING: [Explanation of the comparison and data sources]
        CONFIDENCE: [high/medium/low]
        """
        
        return self._get_llm_response(synthesis_prompt)
    
    def _synthesize_complex_answer(self, query: str, context: str, sub_queries: List[str]) -> Dict:
        """Synthesize answer for complex multi-aspect queries."""
        synthesis_prompt = f"""
        Based on the context from SEC filings, answer this complex financial query.
        
        Original Query: {query}
        Sub-queries analyzed: {', '.join(sub_queries)}
        
        Context:
        {context}
        
        Synthesize a comprehensive answer that addresses all aspects of the original query. Include:
        1. Direct answers to each component
        2. Any calculations or comparisons needed
        3. Overall insights or patterns
        
        Format your response as:
        ANSWER: [Comprehensive answer addressing all query aspects]
        REASONING: [Detailed explanation of analysis and synthesis process]
        CONFIDENCE: [high/medium/low]
        """
        
        return self._get_llm_response(synthesis_prompt)
    
    def _get_llm_response(self, prompt: str) -> Dict:
        """Get structured response from LLM."""
        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1000
                )
            )
            
            content = response.text.strip()
            
            # Parse structured response
            answer_data = {}
            
            # Extract answer
            answer_match = re.search(r'ANSWER:\s*(.+?)(?=REASONING:|$)', content, re.DOTALL)
            if answer_match:
                answer_data["answer"] = answer_match.group(1).strip()
            
            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=CONFIDENCE:|$)', content, re.DOTALL)
            if reasoning_match:
                answer_data["reasoning"] = reasoning_match.group(1).strip()
            
            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*(\w+)', content)
            if confidence_match:
                answer_data["confidence"] = confidence_match.group(1).lower()
            
            # Fallback if parsing fails
            if not answer_data.get("answer"):
                answer_data["answer"] = content
                answer_data["reasoning"] = "Generated from available context"
                answer_data["confidence"] = "medium"
            
            return answer_data
            
        except Exception as e:
            console.print(f"[red]Error in synthesis: {e}[/red]")
            return {
                "answer": "Error occurred during synthesis",
                "reasoning": f"Synthesis error: {str(e)}",
                "confidence": "low"
            }
    
    def _extract_sources(self, retrieval_results: Dict[str, List[Dict]]) -> List[Dict]:
        """Extract source information from retrieval results."""
        sources = []
        seen_sources = set()
        
        for sub_query, results in retrieval_results.items():
            for result in results[:2]:  # Top 2 sources per sub-query
                company = result.get("company", "Unknown")
                year = result.get("year", "Unknown")
                section = result.get("section", "Unknown")
                
                # Create unique identifier for source
                source_id = f"{company}_{year}_{section}"
                
                if source_id not in seen_sources:
                    seen_sources.add(source_id)
                    
                    # Extract meaningful excerpt
                    text = result["text"]
                    excerpt = self._extract_meaningful_excerpt(text, sub_query)
                    
                    sources.append({
                        "company": company,
                        "year": year,
                        "excerpt": excerpt,
                        "section": section,
                        "relevance_score": round(1.0 - result.get("distance", 0.5), 3)
                    })
        
        return sources[:5]  # Limit to top 5 sources
    
    def _extract_meaningful_excerpt(self, text: str, query: str) -> str:
        """Extract a meaningful excerpt from the text relevant to the query."""
        # Simple approach: find sentences containing key terms from query
        query_terms = re.findall(r'\b\w+\b', query.lower())
        sentences = re.split(r'[.!?]+', text)
        
        best_sentence = ""
        max_score = 0
        
        for sentence in sentences:
            if len(sentence) > 50:  # Skip very short sentences
                sentence_lower = sentence.lower()
                score = sum(1 for term in query_terms if term in sentence_lower)
                
                if score > max_score:
                    max_score = score
                    best_sentence = sentence.strip()
        
        if best_sentence:
            # Limit excerpt length
            if len(best_sentence) > 200:
                return best_sentence[:200] + "..."
            return best_sentence
        
        # Fallback: return first meaningful part of text
        return text[:200] + "..." if len(text) > 200 else text

class CalculationEngine:
    """Handles financial calculations and metric computations."""
    
    @staticmethod
    def calculate_growth_rate(old_value: float, new_value: float) -> float:
        """Calculate percentage growth rate."""
        if old_value == 0:
            return 0.0
        return ((new_value - old_value) / old_value) * 100
    
    @staticmethod
    def extract_financial_numbers(text: str) -> List[Dict]:
        """Extract financial numbers from text."""
        numbers = []
        
        # Pattern for financial amounts
        patterns = [
            r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|thousand)?',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|thousand)?\s*dollars?',
            r'(\d+(?:\.\d+)?)\s*%'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1).replace(',', '')
                unit = match.group(2) if len(match.groups()) > 1 else None
                
                try:
                    amount = float(amount_str)
                    
                    # Convert to base units
                    if unit:
                        unit_lower = unit.lower()
                        if unit_lower == 'billion':
                            amount *= 1_000_000_000
                        elif unit_lower == 'million':
                            amount *= 1_000_000
                        elif unit_lower == 'thousand':
                            amount *= 1_000
                    
                    numbers.append({
                        "original_text": match.group(0),
                        "amount": amount,
                        "unit": unit,
                        "position": match.start()
                    })
                except ValueError:
                    continue
        
        return numbers
    
    @staticmethod
    def find_metric_value(text: str, metric: str) -> Optional[float]:
        """Find specific metric value in text."""
        metric_lower = metric.lower()
        text_lower = text.lower()
        
        # Look for the metric followed by a number
        patterns = [
            rf'{re.escape(metric_lower)}[:\s]+\$?(\d+(?:,\d{{3}})*(?:\.\d+)?)\s*(billion|million|thousand)?',
            rf'{re.escape(metric_lower)}[:\s]+(\d+(?:\.\d+)?)\s*%',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    value = float(match.group(1).replace(',', ''))
                    unit = match.group(2) if len(match.groups()) > 1 else None
                    
                    if unit:
                        unit_lower = unit.lower()
                        if unit_lower == 'billion':
                            value *= 1_000_000_000
                        elif unit_lower == 'million':
                            value *= 1_000_000
                        elif unit_lower == 'thousand':
                            value *= 1_000
                    
                    return value
                except (ValueError, IndexError):
                    continue
        
        return None
