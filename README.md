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

Code query outputs:

<img width="867" height="784" alt="Screenshot 2025-09-22 at 12 59 50 AM" src="https://github.com/user-attachments/assets/e281fb4e-ab21-4e2a-ab93-cf68e4cfc220" />

<img width="1629" height="891" alt="Screenshot 2025-09-22 at 9 47 06 AM" src="https://github.com/user-attachments/assets/49677d32-3bc4-4bf9-a35d-22d1e33f0bfa" />

<img width="1619" height="893" alt="Screenshot 2025-09-22 at 9 46 58 AM" src="https://github.com/user-attachments/assets/7ef868f6-aa4d-4414-82d9-317303e1c0cb" />

<img width="1615" height="337" alt="Screenshot 2025-09-22 at 9 44 22 AM" src="https://github.com/user-attachments/assets/a443798f-d695-4782-b775-f086101dd0fb" />

<img width="1617" height="576" alt="Screenshot 2025-09-22 at 9 43 51 AM" src="https://github.com/user-attachments/assets/61ba0856-b9ed-42c5-96d0-05c0c09f7cf8" />

<img width="1623" height="713" alt="Screenshot 2025-09-22 at 9 43 09 AM" src="https://github.com/user-attachments/assets/3aa80065-1e2d-42d1-a039-33f4323ae1c0" />

<img width="1631" height="856" alt="Screenshot 2025-09-22 at 9 43 00 AM" src="https://github.com/user-attachments/assets/ecc72805-53c9-4d13-b1d0-3fc52e69b711" />

<img width="1633" height="662" alt="Screenshot 2025-09-22 at 9 41 21 AM" src="https://github.com/user-attachments/assets/770646ee-1ade-4b61-86cc-2859e7bf50c8" />

<img width="1641" height="631" alt="Screenshot 2025-09-22 at 9 40 35 AM" src="https://github.com/user-attachments/assets/dc621fb0-18f3-4fdc-a344-1fbd3ac072f6" />

<img width="854" height="723" alt="Screenshot 2025-09-22 at 1 00 02 AM" src="https://github.com/user-attachments/assets/831ad8d4-3cf3-4bf3-942f-7468ea1c3ff4" />




