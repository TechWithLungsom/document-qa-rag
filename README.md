## Browser interface
Run these commands from the project folder (with your existing virtual environment activated):

```sh
pip install -r requirements.txt
python -m streamlit run app.py
```

Your browser should open automatically at http://localhost:8501. Upload PDF/TXT files, choose Local AI or Offline mode, then click Process documents. Enter your question and click Find an answer. Expand source cards to inspect page citations. Chunk size, overlap, Top-K and category filtering are available in the interface. The first semantic model load can take several minutes. Keep Ollama running for Local AI mode.

On macOS, you can also double-click `Start UI.command` to install dependencies and launch the interface. Keep its Terminal window open while using the app.

# Document Question Answering with RAG

## Run
Python 3.11 recommended. From this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python rag.py
python evaluate.py
```

The default is an offline, extractive baseline. For semantic retrieval and generated answers, install Ollama, download a model with `ollama pull llama3.2:3b`, start Ollama, then run:

```sh
python rag.py --semantic --model llama3.2:3b --question "How long is probation?" --k 3
python rag.py --semantic --model llama3.2:3b --category hr --size 500 --overlap 80
```

Add PDFs to data/. PDF ingestion uses PyPDF and retains one-based page numbers. TXT fixtures use form feed for page boundaries. Empty pages and unreadable documents are counted/reported; scanned PDFs require external OCR. Category is the filename prefix before the first underscore. No secrets are required. The CLI is the assignment's permitted interface.

## Architecture and choices
Files → page extraction → configurable overlapping character windows → embeddings → vector index → category-filtered Top-K retrieval → Ollama → answer and retrieved source/page evidence.

Production mode uses Sentence Transformers all-MiniLM-L6-v2 and FAISS inner product over normalized embeddings (cosine similarity). Embeddings represent semantic relationships so differently worded passages can match a query. SQLite persists vectors and metadata for inspection; each CLI run rebuilds its search index. The dependency-free baseline uses normalized hashed word counts and exact cosine scanning, making experiments reproducible without network downloads. It is lexical, not a semantic substitute. The simple character splitter is an alternative to LangChain: it has explicit reproducible boundaries and overlap, though it can cut sentences. All retrieval results include text, source, page, category, score and chunk ID.

## Deliverables
- rag.py: ingestion, chunking, indexing, retrieval, Ollama and CLI.
- evaluate.py: 12 labelled questions and 48 observed baseline runs.
- data/: original, shareable fictional policies as paginated TXT fixtures.
- results/evaluation.json: full passages, labels, retrieval metrics and extractive answers.
- results/filter.json: metadata experiment.
- results/demo.txt: captured CLI demonstration (ingestion, context and citations).
- REPORT.md: findings and conceptual answers.

## Limits and verification
Offline CLI and evaluation were executed successfully. Semantic dependencies, model download and live Ollama generation were not executed in the delivery environment. No LLM answer-quality score is claimed. Citations in generated prose are model-produced; the displayed retrieval list is authoritative evidence provenance, not proof that every generated claim is supported. Review answers before relying on them. Local documents are sent only to the local Ollama endpoint. The fixtures contain fictional policies.
