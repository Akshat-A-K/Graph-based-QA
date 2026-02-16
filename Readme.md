# DRG-Doc-QA

**Framework for Externalized Graph-Based Reasoning for Faithful Multilingual Question Answering over Documents**

---

## 📌 Overview

This project implements a **Document Reasoning Graph (DRG)** framework for faithful document question answering.
Instead of letting an LLM reason internally (which often causes hallucinations), reasoning is **externalized into an explicit graph** built from the document.

Pipeline:

```
PDF → sentence nodes → graph (DRG)
     → query grounding → graph reasoning
     → evidence nodes → answer generation
```

The system is designed for:

* rule-heavy documents (policies, manuals, guidelines)
* interpretable reasoning
* multilingual / paraphrased queries
* faithful answers grounded in document text

---

## 🧠 Key Idea

Large Language Models often:

* hallucinate
* ignore exceptions
* miss constraints

We fix this by:

1. Converting the document into a graph
2. Running reasoning on the graph
3. Sending only verified evidence to the LLM

So the LLM **writes the answer** but does **not perform reasoning**.

---

## 🏗 Project Structure

```
inlp_project/
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
