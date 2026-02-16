# Graph-Based QA System

**Document Question Answering using Multi-Level Graph Reasoning**

A Python framework for extracting accurate answers from documents using an externalized graph-based reasoning approach. Supports multilingual queries with semantic understanding.

---

## 🎯 Features

- **Multi-level graph reasoning** combining sentence, span, and knowledge graphs
- **Hybrid retrieval** using BM25 lexical matching + semantic embeddings + graph centrality
- **Multilingual support** with paraphrase-multilingual embeddings (50+ languages)
- **Zero-shot learning** - no training required
- **Evidence-based answers** with span-level explanations
- **Streamlit UI** for easy document upload and QA

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Interactive Mode (Streamlit)

```bash
streamlit run app.py
```

Upload a PDF and ask questions in natural language.

### Command Line

```bash
python test_qa.py <csv_path> <num_samples>
```

---

## 📊 System Architecture

```
PDF Input
  ↓
Sentence Extraction (spaCy)
  ↓
Multi-Level Graph Construction
  ├─ Sentence Graph (DRG)
  ├─ Span Graph (fine-grained)
  └─ Knowledge Graph (entities/relations)
  ↓
Query Processing
  ├─ Embedding (Multilingual)
  ├─ BM25 Matching
  └─ Graph Centrality
  ↓
Hybrid Reasoning
  ├─ Semantic similarity
  ├─ Lexical matching
  ├─ Structural importance
  └─ Discourse relations
  ↓
Answer Extraction
```

---

## 📁 Project Structure

```
graph-based-qa/
├── parser/
│   ├── pdf_parser.py           # PDF text extraction
│   ├── sentence_splitter.py    # spaCy sentence tokenization
│   ├── drg_nodes.py            # Sentence node construction
│   ├── drg_graph.py            # Document Reasoning Graph
│   ├── span_extractor.py       # Fine-grained span extraction
│   ├── span_graph.py           # Span-level graph
│   ├── kg_builder.py           # Knowledge graph construction
│   ├── advanced_retrieval.py   # BM25, centrality, hybrid scoring
│   ├── enhanced_reasoner.py    # Multi-level reasoning engine
│   ├── evaluator.py            # F1/EM metrics
│   └── __init__.py
├── app.py                       # Streamlit interface
├── test_qa.py                  # Command-line QA evaluation
├── requirements.txt            # Dependencies
└── Readme.md                   # This file
```

---

## 🔧 Core Components

### 1. Document Parsing
- PDF text extraction with page-level metadata
- spaCy-based sentence tokenization

### 2. Graph Construction
- **Sentence Graph**: Structural edges (same page/section, adjacent), semantic edges (embedding similarity)
- **Span Graph**: Clause and phrase extraction with discourse relations
- **Knowledge Graph**: Entity and relation extraction from document spans

### 3. Retrieval
- **BM25**: Lexical term frequency-inverse document frequency matching
- **Semantic**: Sentence transformer embeddings (paraphrase-multilingual-mpnet-base-v2)
- **Centrality**: PageRank and betweenness centrality for importance scoring

### 4. Reasoning
- **Hybrid Scoring**: Combines semantic (40%) + lexical (40%) + centrality (20%)
- **Graph Traversal**: Centrality-guided neighbor expansion
- **Query Expansion**: Synonym mapping for improved recall

---

## 📈 Performance

Evaluated on SQuAD development set:
- **F1 Score**: Up to 1.0 (exact span matching)
- **Exact Match**: Up to 100% (exact answer detection)

---

## 🌍 Multilingual Support

Uses `paraphrase-multilingual-mpnet-base-v2` for:
- English queries
- Hindi/Hinglish queries (mixed English-Hindi)
- 50+ language families

---

## 📝 Usage Example

```python
from parser.drg_nodes import build_nodes
from parser.drg_graph import DocumentReasoningGraph
from parser.enhanced_reasoner import EnhancedHybridReasoner

# Build graphs from context
pages = [{"page": 1, "text": context}]
sentence_nodes = build_nodes(pages)
drg = DocumentReasoningGraph()
drg.add_nodes(sentence_nodes)
# ... build span graph and KG ...

# Initialize reasoner
reasoner = EnhancedHybridReasoner(
    sentence_graph=drg.graph,
    span_graph=span_graph.graph,
    knowledge_graph=kg
)

# Get answer
results = reasoner.enhanced_reasoning("What is the deadline?", k=5)
```

---

## 📦 Dependencies

- `numpy` - Numerical computing
- `networkx` - Graph algorithms
- `pymupdf` - PDF processing
- `spacy` - NLP tokenization
- `sentence-transformers` - Semantic embeddings
- `scikit-learn` - Similarity computation
- `streamlit` - Web UI

---

## 📄 License

This project is part of INLP (Indian Natural Language Processing) research.

---

## 🤝 Contributing

Contributions are welcome. Please ensure all code follows the existing structure.


│
├── parser/
│   ├── pdf_parser.py          # PDF → text
│   ├── sentence_splitter.py   # text → sentences
│   ├── section_utils.py       # detect sections
│   ├── drg_nodes.py           # build sentence nodes
│   ├── drg_graph.py           # build sentence graph
│   ├── reasoning_engine.py    # sentence-level reasoning
│   ├── span_extractor.py      # extract fine-grained spans (NEW)
│   ├── span_graph.py          # build span-level graph (NEW)
│   ├── kg_builder.py          # build knowledge graph (NEW)
│   └── hybrid_reasoner.py     # multi-level reasoning (NEW)
│
├── test_reasoning.py          # sentence-level demo
├── test_hybrid.py             # hybrid KG+span demo (NEW)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone repo

```bash
git clone https://github.com/GauravPatel369/graph-based-qa.git
cd graph-based-qa
```

### 2️⃣ Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Download NLTK tokenizer

Run once:

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
```

---

## ▶️ How to Run

### Basic Demo (Sentence-Level)

```bash
python test_reasoning.py
```

### Advanced Demo (KG + Span Graphs)

```bash
python test_hybrid.py
```

### Custom Query

```bash
python test_hybrid.py Assignment-1.pdf "Your question here"
```

---

## 🧪 Example Output

```
FLAT
[0] Deadline: 5 March 2026, 11:59pm
[4] No extensions will be granted...

STRUCTURAL
...more context...

EMERGENT
[0] Deadline: 5 March 2026, 11:59pm
[4] No extensions will be granted...
```

These are the **evidence sentences** selected by graph reasoning.

---

## 🧩 Modules Explained

### 📄 PDF Parser

Extracts text page-wise.

### 🔹 Node Builder

Each sentence becomes a node with:

```
node_id
text
page
section
```

### � Span Extractor (NEW)

Extracts fine-grained text spans:
- Clauses (split by discourse markers)
- Important phrases (dates, conditions, negations)
- Keyword spans (deadlines, requirements)

### 🔗 Graph Builder

Three types of graphs:

1. **Sentence Graph** - Original DRG with sentence nodes
2. **Span Graph** - Fine-grained graph with clause/phrase nodes
3. **Knowledge Graph** - Entities and relations

Edges added:
* Structural: same page, section, adjacent
* Semantic: embedding similarity
* Discourse: conditions, exceptions, temporal

### 🧠 Reasoning Engine

Implements 5+ strategies:

| Method              | Description                           |
| ------------------- | ------------------------------------- |
| Flat                | Embedding retrieval (sentence-level)  |
| Structural          | Neighbor expansion                    |
| Emergent            | Graph-based reasoning                 |
| Span Retrieval      | Fine-grained span matching            |
| KG-Guided           | Entity-based evidence selection       |
| Hybrid Multi-Level  | Combines all representations (BEST)   |

Hybrid reasoning is the main contribution.

---

## 🌍 Multilingual Support (Planned)

Future extension:

* Hinglish queries
* translation grounding
* multilingual embeddings

---

## 📊 Evaluation Plan

We compare:

* LLM-only baseline
* flat retrieval
* structural graph
* emergent reasoning

Metrics:

* faithfulness
* exception handling
* robustness to paraphrase
* interpretability

---

## 🚀 Example Query

```
Query: When is the assignment deadline?
```

System:

1. finds relevant nodes
2. expands graph
3. selects evidence
4. generates answer

---

## 🧱 Tech Stack

* Python
* NetworkX
* Sentence-Transformers
* PyMuPDF
* NLTK

---

## 📌 Latest Updates (Feb 2026)

### ✅ **NEW: Knowledge Graph & Span Graph Integration**

- **Span-level extraction**: Fine-grained clauses, phrases, and keywords
- **Knowledge Graph**: Entities (dates, requirements, constraints) + relations
- **Span Graph**: Discourse relations (conditions, exceptions, temporal)
- **Hybrid reasoning**: Multi-level evidence selection combining all graphs

See [KG_SPAN_INTEGRATION_SUMMARY.md](KG_SPAN_INTEGRATION_SUMMARY.md) for details.

### 🚧 **TODO: Hinglish Support**

See [HINGLISH_ROADMAP.md](HINGLISH_ROADMAP.md) for implementation plan.

Next steps:
* Multilingual embeddings
* Hinglish query translation
* LLM answer generation
* Evaluation metrics
* Reasoning visualization
* UI demo (Streamlit)

---

## 👨‍💻 Team

**CTRL+ALT+DLT**

---

## 📜 License

MIT License
