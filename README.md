# Financial Q&A System

A RAG-based system with AI agent capabilities for answering complex financial questions about Google, Microsoft, and NVIDIA using their SEC 10-K filings (2022-2024).

## Features

- **Intelligent Query Processing**: Automatically classifies and decomposes complex queries
- **Multi-step Reasoning**: Agent-based architecture for comparative and cross-company analysis
- **Vector Search**: ChromaDB with Google embeddings for semantic retrieval
- **Financial Context**: Specialized chunking and parsing for SEC documents

## Quick Start

1. **Install**: `pip install -r requirements.txt`
2. **Setup Environment**: Create `.env` with `GOOGLE_API_KEY=your_key`
3. **Initialize System**: `python main.py setup`
4. **Ask Questions**: `python main.py query "What was Microsoft's revenue in 2023?"`

## Example Queries

- Simple: "What was NVIDIA's total revenue in 2024?"
- Comparative: "How did Google's cloud revenue grow from 2022 to 2023?"
- Cross-company: "Which company had the highest operating margin in 2023?"

Built with Google Gemini, ChromaDB, and specialized financial document processing.

Sample output can be found here: https://github.com/mananchopra/financial-q-a-system/blob/main/sample_outputs.md or https://github.com/mananchopra/financial-q-a-system/blob/main/sample_outputs.json

Demo script to run all queries: https://github.com/mananchopra/financial-q-a-system/blob/main/run_all_queries.py

Brief Design Doc: https://github.com/mananchopra/financial-q-a-system/blob/main/DESIGN.md
