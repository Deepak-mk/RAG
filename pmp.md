Phase 1: Project Initialization & Environment Setup
Goal: Establish the foundational Python environment and API keys.
Initialize the Project: Use uv init . in your project folder to create a new Python workspace
.
Install Dependencies: Run uv add to install the required libraries: fastapi, ingest, llama-index-core, llama-index-readers-file, python-dotenv, qdrant-client, uvicorn, streamlit, and openai
.
Environment Variables: Create a .env file in the root directory and add your OPENAI_API_KEY to authenticate with OpenAI services
.
Phase 2: Data Pipeline & Vector Database Setup
Goal: Configure the storage and parsing logic for handling PDF documents.
Launch Qdrant (Database): Install Docker Desktop and run the Qdrant container locally on port 6333, mapping a local volume (e.g., /qdrant_storage) for persistent data
.
Database Client (vector_db.py):
Create a QuadrantStorage class to initialize a collection named "documents" using cosine distance and 372 dimensions
.
Write an upsert method to insert vectors and payloads
.
Write a search method to query the database and return the top k relevant text snippets
.
Data Loader (data_loader.py):
Use Llama Index’s SentenceSplitter to parse PDFs into chunks (size 1,000, overlap 200)
.
Implement an embed_text function using OpenAI's text-embedding-3-large model to convert the text chunks into 372-dimensional vectors
.
Phase 3: Core Application & Orchestration Logic
Goal: Build the backend API and wire up Ingest for step-by-step observability.
Define Custom Types (customtypes.py): Create Pydantic base models (e.g., RagChunkAndSrc, RagUpsertResult, RagSearchResult) to strictly define your data structures
.
Initialize FastAPI & Ingest (main.py): Set up the FastAPI app and bind it to the Ingest client (ingest.fast_api.serve) while keeping is_production=False for local development
.
Build Ingestion Workflow (rag_ingest_pdf):
Create an Ingest function triggered by the rag/ingest_pdf event
.
Step 1: Call the load_and_chunk_pdf function to extract text
.
Step 2: Generate embeddings and upsert them into the Qdrant database, logging the operation as a tracked step
.
Build Query Workflow (rag_query_pdf_ai):
Create a second Ingest function triggered by rag/query_pdf_ai
.
Step 1: Embed the user's question and search the Qdrant database for the top 5 matches
.
Step 2: Construct a prompt and use Ingest's AI adapter (ctx.step.ai.infer) to pass the context and question to the gpt-4o-mini model, defining parameters like a temperature of 0.2 and max tokens of 1024
.
Phase 4: Local Testing & UI Integration
Goal: Connect the backend to a user-friendly frontend interface.
Start Local Servers:
Run the backend using uv run uvicorn main:app
.
Run the Ingest development server using Node.js: npx ingest-cli@latest dev -u http://127.0.0.1:8000/api/ingest
.
Frontend Development (streamlit_app.py): Build a Streamlit interface that allows users to upload PDFs and submit questions
.
Event Polling: Configure the Streamlit app to trigger Ingest events and repeatedly poll the Ingest server's API to fetch the completed runs and display the LLM's final answer
.
Phase 5: Production Readiness
Goal: Secure the application and implement guardrails for the wild.
Flow Control: Apply Ingest's flow control decorators directly to your functions to add features like rate limiting (e.g., only allowing ingestion of a specific PDF source ID once every 4 hours) and throttling
.
Deployment Prep: Transition the app state by setting is_production=True in your Ingest client and configuring event keys for secure communication between your infrastructure and Ingest's managed servers