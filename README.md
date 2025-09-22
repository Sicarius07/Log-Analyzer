The only challenging part was filtering the structured logs based on the natural language user prompt. The later steps, analyzing the filtered logs and identifying the most relevant ones were comparatively straightforward and could be handled via standard LLM calls.

For filtering, I considered multiple approaches. I implemented a multi-step, tool-call based filtering over a pandas DataFrame. While this in-memory approach worked well for a small use case, a more scalable solution would be to index and filter logs using Elasticsearch or a similar system.

During experimentation, I also noticed a high number of near-duplicate logs, we could have done clustering. Also, this negatively impacted the accuracy of vector semantic search. A useful enhancement here would be applying clustering  before indexing into the vector database. Although the initial indexing step would be slower, subsequent analysis and retrieval would become significantly faster and more precise.

And for calculating cost I have hardcoded the token cost for gpt-4.1-mini. And used just prompt and completion tokens for cost calculation

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
