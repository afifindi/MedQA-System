# Medical Question Answering System

A production-ready Medical Question Answering System using Retrieval-Augmented Generation (RAG) with Gemma-2B-IT (LoRA fine-tuned) and FAISS-based semantic retrieval.

## Architecture

This system uses a modular clean architecture:
- **FastAPI Backend**: Orchestrates the RAG pipeline.
- **FAISS**: Vector database for fast semantic retrieval.
- **SentenceTransformers**: Dense embedding model for query encoding.
- **Gemma-2B-IT + LoRA**: Fine-tuned LLM for generating medical answers.
- **React + Vite Frontend** (To be implemented).

## Quick Start (Backend)

1. Clone the repository.
2. Ensure you have the fine-tuned model artifacts in `models/gemma_medical_qa_final`.
3. Set up a Python 3.11 virtual environment and install dependencies:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` in the `backend` directory and adjust if needed.
5. Run the FastAPI development server:
   ```bash
   python main.py
   ```
   Or use uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
