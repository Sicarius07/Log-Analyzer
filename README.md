## Backend Implementation
- **DataFrame Processing**: Flattens NDJSON logs into pandas DataFrame with UUIDs
- **Vector Indexing**: ChromaDB for semantic search
- **Structured Filtering**: User query -> LLM analyzes columns -> extracts unique values -> creates hierarchical filters
- **Hybrid Search**: Combines structured DataFrame filtering + semantic vector search
- **Final Analysis**: GPT-4.1-mini analyzes filtered logs with structured output 


## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- OpenAI API key

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
cp env_example.txt .env
# Edit .env and add your OPENAI_API_KEY
python main.py
```
Backend runs on: http://localhost:8000

### Frontend Setup
```bash
cd frontend
npm install
npm start
```
Frontend runs on: http://localhost:3000

### Usage
1. Upload NDJSON log file (automatic indexing with progress bar)
2. Describe the incident/issue
3. Click "Analyze Logs (Fast)" for analysis
4. View filtered logs with expandable JSON and markdown analysis

## Architecture
- **Backend**: FastAPI + ChromaDB + OpenAI + Pandas
- **Frontend**: React + Shadcn/ui + Tailwind CSS (dark theme)
- **LLM**: GPT-4.1-mini



The primary challenge was filtering structured logs based on natural language user prompts. Subsequent steps—such as analyzing the filtered logs and identifying the most relevant entries—were comparatively straightforward and could be effectively handled through standard LLM calls.

For the filtering process, multiple approaches were explored. A multi-step, tool-call–based filtering mechanism was implemented using a pandas DataFrame. While this in-memory method performed well for smaller datasets, a more scalable solution would involve indexing and querying logs using Elasticsearch or a similar search system.

During experimentation, a high number of near-duplicate logs were observed, which adversely affected the accuracy of vector semantic search. Applying clustering before indexing into the vector database could significantly improve performance—although it would increase the initial indexing time, subsequent analysis and retrieval would become faster and more precise.

For cost estimation, the token cost for gpt-4.1-mini is hardcoded. 
