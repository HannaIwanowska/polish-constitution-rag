# Polish Constitution RAG System

A Retrieval-Augmented Generation (RAG) application that answers questions
about the Polish Constitution with article-level source citations.



## 🏗️ Architecture

### Document Ingestion (`src/1_index_pipeline.py`)

Instead of using a default text splitter, the pipeline applies regex-based
parsing that splits the document on constitutional article boundaries
(`Art. \d+`). Each chunk carries metadata (`rozdzial`, `artykul`), allowing
the system to cite exact sources in responses (e.g., *"Art. 154 ust. 1"*).

### Query Pipeline (`src/2_query_pipeline.py`)

A LangChain LCEL chain composed of:
1. **Retriever** (top-k=15) over a ChromaDB vector store.
2. **Document formatter** that prefixes context with article numbers.
3. **Strict prompt** instructing the LLM to:
 - Answer only based on retrieved context.
 - Always cite the article number.
 - Refuse with a fixed phrase if the answer is out-of-scope.
4. **LLM:** GPT-4o (`temperature=0.0`) for deterministic legal answers.

### Tech Stack

| Component | Choice |
|-----------|--------|
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector DB | ChromaDB |
| Framework | LangChain (LCEL) |
| Language | Python 3.10+ |

> **Why OpenAI over a local model?** The project started with a local
> Bielik-7B + Hugging Face embeddings setup, but response quality on legal
> Polish text was significantly lower than the OpenAI stack. 



## 🧪 Example Outputs

```
❓ Kto powołuje Prezesa Rady Ministrów?
✅ Prezydent Rzeczypospolitej desygnuje Prezesa Rady Ministrów. (Art. 154 ust. 1)

❓ Ile lat trwa kadencja Sejmu i Senatu?
✅ Kadencja Sejmu i Senatu trwa cztery lata. (Art. 98 ust. 1)

❓ Jak upiec idealną szarlotkę?  ← out-of-scope
✅ Nie znalazłem odpowiedzi w Konstytucji.
```


## 🚀 How to Run

```bash
# 1. Clone and enter the repo
git clone https://github.com/HannaIwanowska/polish-constitution-rag.git
cd polish-constitution-rag

# 2. Set up environment
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 3. Configure API key
cp .env.example .env
# Then paste your OpenAI key into .env

# 4. Build the vector store (one-time)
python src/1_index_pipeline.py

# 5. Run example queries
python src/2_query_pipeline.py
```


## ⚠️ Scope & Limitations

This project was developed as part of postgraduate studies in AI Engineering,
with the goal of hands-on learning of modern RAG architecture. As such, it is
a working prototype rather than a production-ready system.

**Current scope:**
- Single-document indexing (Polish Constitution).
- Local CLI execution.
- Manual evaluation against a fixed set of test questions.

**Areas I plan to explore next:**
- **API layer (FastAPI)** – wrapping the system as a REST service is the
most natural next step, and is part of my ongoing coursework.
- **Automated evaluation** – exploring frameworks such as RAGAS to move
from manual quality checks to measurable metrics.
- **Retrieval improvements** – experimenting with hybrid search
(semantic + keyword) for better handling of specific legal terms.