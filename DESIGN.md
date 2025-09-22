# Financial Q&A System - Design Document

## System Architecture Overview

The Financial Q&A System implements a **Retrieval-Augmented Generation (RAG)** architecture enhanced with **AI agent capabilities** for intelligent query processing and multi-step reasoning.

## 1. Chunking Strategy

### **Approach: Financial-Aware Semantic Chunking**

**Configuration:**
- Chunk size: 700 tokens (~2,800 characters)
- Overlap: 100 tokens (14% overlap for context preservation)
- Token estimation: Character-based approximation (1 token ≈ 4 characters)

**Key Features:**
1. **Semantic Boundary Preservation**: Chunks end at sentence boundaries to maintain context integrity
2. **Financial Pattern Detection**: Identifies and preserves financial metrics, percentages, and monetary amounts
3. **Section-Aware Processing**: Maintains SEC filing structure (Item 1, 1A, 7, 8, etc.)
4. **Metadata Enrichment**: Each chunk includes company, year, section, and financial content scoring

**Rationale:**
- 700 tokens balances context richness with embedding model limits
- Overlap ensures no critical information is lost at chunk boundaries
- Financial pattern awareness prevents splitting of monetary figures and metrics
- Section awareness enables targeted retrieval for specific filing areas

## 2. Embedding Model Choice

### **Selected Model: Google Text-Embedding-004**

**Specifications:**
- Dimensions: 768
- Context window: 2,048 tokens
- Task-specific optimization: Separate embeddings for documents vs. queries

**Why This Choice:**

1. **Performance**: State-of-the-art retrieval performance on financial documents
2. **Integration**: Seamless integration with Google Gemini LLM ecosystem
3. **Task Specialization**: Optimized embeddings for `retrieval_document` and `retrieval_query` tasks
4. **Cost Efficiency**: Competitive pricing for high-volume embedding generation
5. **Proven Results**: Excellent semantic understanding of financial terminology and concepts

**Alternative Considered:**
- OpenAI text-embedding-3-small: Good performance but less integrated with our Gemini-based reasoning pipeline

## 3. Agent/Query Decomposition Approach

### **Multi-Layer Agent Architecture**

#### **Query Classification (5 Types):**
1. **SIMPLE_DIRECT**: Single metric, single company (`"What was Microsoft's revenue in 2023?"`)
2. **COMPARATIVE_YOY**: Year-over-year comparisons (`"How did NVIDIA grow from 2022 to 2023?"`)
3. **CROSS_COMPANY**: Multi-company analysis (`"Which company has highest margins?"`)
4. **COMPLEX_MULTI_ASPECT**: Multi-step calculations (`"Compare R&D ratios across companies"`)
5. **SEGMENT_ANALYSIS**: Business segment breakdowns (`"What % of Google revenue is ads?"`)

#### **Classification Methods:**
1. **Pattern Matching**: Regex-based initial classification for common patterns
2. **Entity Extraction**: Automatic detection of companies, years, financial metrics
3. **LLM Classification**: Gemini-powered classification for complex or ambiguous queries
4. **Complexity Scoring**: Multi-dimensional scoring based on entities and operations required

#### **Query Decomposition Strategies:**

**Simple Queries**: No decomposition - direct retrieval

**Comparative Queries**: Temporal decomposition
```
"NVIDIA growth 2022-2023" → ["NVIDIA revenue 2022", "NVIDIA revenue 2023"]
```

**Cross-Company Queries**: Entity-based decomposition  
```
"Best margins 2023" → ["Google margin 2023", "Microsoft margin 2023", "NVIDIA margin 2023"]
```

**Complex Queries**: LLM-powered decomposition
```
"R&D ratios comparison" → ["Company X R&D 2023", "Company X revenue 2023", ...]
```

#### **Retrieval Strategy Selection:**
- **Semantic Search**: Default vector similarity
- **Company-Focused**: Metadata filtering by company for cross-company queries
- **Temporal**: Time-based filtering for year-over-year analysis
- **Hybrid**: Combined semantic + metadata filtering for complex queries

#### **Synthesis Process:**
1. **Context Aggregation**: Combine all retrieval results with metadata
2. **LLM Reasoning**: Gemini-powered analysis and calculation
3. **Answer Generation**: Structured response with reasoning chain
4. **Source Attribution**: Link answers back to specific SEC filing excerpts
5. **Confidence Assessment**: Scoring based on retrieval quality and answer clarity

### **Key Design Decisions:**

1. **Agent-First Architecture**: Query understanding before retrieval improves accuracy vs. naive RAG
2. **Adaptive Retrieval**: Different strategies for different query types vs. one-size-fits-all
3. **Multi-Step Reasoning**: Explicit decomposition enables complex financial analysis
4. **Source Traceability**: Every answer links back to specific SEC filing sections
5. **Confidence Scoring**: Transparent assessment of answer reliability

This design enables the system to handle everything from simple lookups to complex multi-company financial analysis while maintaining accuracy and traceability to source documents.