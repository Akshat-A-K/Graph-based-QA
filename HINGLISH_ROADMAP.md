# Roadmap: Hinglish QA Integration

## 🎯 Goal
Transform the current English-only DRG system into a **Hinglish Code-Mixed Question Answering System** aligned with your INLP project proposal.

---

## 📋 Current Gap Analysis

| Component | Current State | Required for Project | Priority |
|-----------|--------------|---------------------|----------|
| Document Language | English PDFs | English (OK) | ✅ |
| Query Language | English only | **Hinglish/Code-mixed** | 🔴 HIGH |
| Embeddings | English-only model | Multilingual | 🔴 HIGH |
| Query Grounding | Basic similarity | Paraphrase-robust | 🟡 MEDIUM |
| Answer Generation | Missing | LLM-based | 🟡 MEDIUM |
| Evaluation | Missing | Faithfulness metrics | 🟢 LOW |

---

## 🚀 Phase 1: Multilingual Embeddings (Week 1)

### Objective
Enable the system to understand Hinglish queries like:
- "Assignment ki deadline kab hai?"
- "Extension milega kya?"
- "Submission ke liye kya chahiye?"

### Implementation

#### Step 1.1: Replace Embedding Model
**File:** [parser/drg_graph.py](parser/drg_graph.py), [parser/span_graph.py](parser/span_graph.py), [parser/hybrid_reasoner.py](parser/hybrid_reasoner.py)

```python
# OLD (English-only)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# NEW (Multilingual)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# OR (Indic-focused)
model = SentenceTransformer("l3cube-pune/indic-sentence-bert-nli")
```

**Testing:**
```bash
python test_hybrid.py Assignment-1.pdf "Deadline kab hai?"
python test_hybrid.py Assignment-1.pdf "Extension nahi milega kya?"
```

#### Step 1.2: Create Hinglish Query Test Set
**File:** `test_data/hinglish_queries.json`

```json
[
  {
    "query_en": "When is the assignment deadline?",
    "query_hi": "Assignment ki deadline kab hai?",
    "expected_spans": ["Deadline: 11th February 2026"]
  },
  {
    "query_en": "Will extensions be granted?",
    "query_hi": "Extension milega kya?",
    "expected_spans": ["no extension will be possible"]
  },
  {
    "query_en": "What file format should I submit?",
    "query_hi": "Kaunsa file format submit karna hai?",
    "expected_spans": [".zip file needs to be uploaded"]
  }
]
```

#### Step 1.3: Benchmark Script
**File:** `evaluate_hinglish.py`

```python
import json
from parser.pdf_parser import extract_pages
from parser.drg_nodes import build_nodes
from parser.span_extractor import build_span_nodes
from parser.span_graph import build_span_graph
from parser.kg_builder import build_knowledge_graph
from parser.hybrid_reasoner import HybridReasoner

# Load test queries
with open("test_data/hinglish_queries.json") as f:
    queries = json.load(f)

# Build graphs once
pages = extract_pages("Assignment-1.pdf")
sentence_nodes = build_nodes(pages)
# ... (build all graphs)

reasoner = HybridReasoner(sentence_graph, span_graph, kg)

# Test each query
for item in queries:
    for lang in ["query_en", "query_hi"]:
        query = item[lang]
        results = reasoner.hybrid_reasoning(query)
        # Check if expected spans are retrieved
        print(f"Query ({lang}): {query}")
        print(f"Top spans: {results['span_results'][:3]}")
        print()
```

---

## 🚀 Phase 2: Query Translation/Grounding (Week 2)

### Objective
Handle queries that use different words than the document:
- Query: "Due date kya hai?" → Document: "deadline"
- Query: "Extension possible hai?" → Document: "no extension will be possible"

### Option A: Translation Pipeline (Simpler)

**File:** `parser/query_translator.py`

```python
from googletrans import Translator  # or indic-trans library

class QueryTranslator:
    def __init__(self):
        self.translator = Translator()
    
    def translate_to_english(self, query: str) -> str:
        """Translate Hinglish/Hindi to English."""
        # Detect if query has Hindi script
        if any('\u0900' <= c <= '\u097F' for c in query):
            result = self.translator.translate(query, src='hi', dest='en')
            return result.text
        return query
    
    def expand_query(self, query: str) -> list:
        """Generate query variations."""
        # Example: "deadline" → ["deadline", "due date", "submission date"]
        synonyms = {
            "deadline": ["deadline", "due date", "submission date", "last date"],
            "extension": ["extension", "extra time", "additional time"],
            "submit": ["submit", "upload", "turn in", "hand in"]
        }
        
        expanded = [query]
        for word, syns in synonyms.items():
            if word in query.lower():
                for syn in syns:
                    expanded.append(query.lower().replace(word, syn))
        
        return expanded
```

**Integration:**
```python
# In hybrid_reasoner.py
def hybrid_reasoning(self, query: str, k: int = 10):
    # Translate if needed
    translator = QueryTranslator()
    query_en = translator.translate_to_english(query)
    
    # Generate variations
    query_variations = translator.expand_query(query_en)
    
    # Retrieve using all variations
    all_results = []
    for q in query_variations:
        results = self._retrieve_internal(q)
        all_results.extend(results)
    
    # Deduplicate and rank
    return self._merge_results(all_results)
```

### Option B: Cross-lingual Embeddings (Better)

Already done if you use multilingual SBERT! The embeddings handle translation automatically.

---

## 🚀 Phase 3: LLM Answer Generation (Week 3)

### Objective
Convert evidence spans into natural language answers.

**File:** `parser/answer_generator.py`

```python
import openai  # or use local Llama/Gemma

class AnswerGenerator:
    def __init__(self, model="gpt-4", use_local=False):
        self.model = model
        self.use_local = use_local
    
    def generate_answer(self, query: str, evidence_spans: list, query_language="en") -> str:
        """Generate answer from evidence spans."""
        
        # Format evidence
        evidence_text = "\n".join([
            f"{i+1}. {span['text']}"
            for i, span in enumerate(evidence_spans)
        ])
        
        # Determine response language
        lang_instruction = ""
        if query_language == "hi":
            lang_instruction = "Respond in Hinglish (Hindi-English mix)."
        
        prompt = f"""You are a helpful assistant answering questions based on a document.

INSTRUCTIONS:
- Answer ONLY based on the evidence provided below
- If evidence doesn't contain the answer, say "Information not available in document"
- Be concise and accurate
- {lang_instruction}

EVIDENCE:
{evidence_text}

QUESTION: {query}

ANSWER:"""
        
        if self.use_local:
            # Use local model (Llama, Gemma, etc.)
            return self._generate_local(prompt)
        else:
            # Use OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.3
            )
            return response.choices[0].message.content
    
    def _generate_local(self, prompt: str) -> str:
        # TODO: Integrate local LLM
        # Example with transformers:
        # from transformers import pipeline
        # generator = pipeline('text-generation', model='meta-llama/Llama-2-7b-chat-hf')
        # return generator(prompt)[0]['generated_text']
        pass
```

**Integration:**
```python
# In test_hybrid.py or new test_qa.py

from parser.answer_generator import AnswerGenerator

# After getting evidence spans
hybrid_results = reasoner.hybrid_reasoning(query)
evidence_spans = [
    {"text": span_graph.nodes[sid]["text"]}
    for sid in hybrid_results["span_results"]
]

# Generate answer
generator = AnswerGenerator(model="gpt-4")
answer = generator.generate_answer(query, evidence_spans)

print(f"\nQUESTION: {query}")
print(f"ANSWER: {answer}")
```

---

## 🚀 Phase 4: Evaluation Framework (Week 4)

### Objective
Measure system performance on key metrics from your proposal.

**File:** `evaluation/evaluator.py`

```python
class QAEvaluator:
    def __init__(self):
        pass
    
    def evaluate_faithfulness(self, answer: str, evidence_spans: list) -> float:
        """
        Check if answer is grounded in evidence.
        Returns score 0-1.
        """
        # Simple: check if answer phrases appear in evidence
        answer_lower = answer.lower()
        
        matched_words = 0
        total_words = len(answer.split())
        
        for span in evidence_spans:
            span_text = span["text"].lower()
            for word in answer.split():
                if word in span_text and len(word) > 3:
                    matched_words += 1
        
        return matched_words / max(total_words, 1)
    
    def evaluate_robustness(self, queries: list, expected_spans: list) -> dict:
        """
        Test with paraphrased queries.
        Returns retrieval consistency score.
        """
        # queries = ["When is deadline?", "What is due date?", "Deadline kab hai?"]
        # Check if all queries retrieve the same core evidence
        
        results = []
        for query in queries:
            spans = self.reasoner.hybrid_reasoning(query)["span_results"]
            results.append(set(spans[:5]))  # top 5
        
        # Compute overlap (Jaccard similarity)
        overlap_scores = []
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                overlap = len(results[i] & results[j]) / len(results[i] | results[j])
                overlap_scores.append(overlap)
        
        return {
            "avg_overlap": sum(overlap_scores) / len(overlap_scores),
            "details": overlap_scores
        }
    
    def evaluate_exception_handling(self, query: str, evidence_spans: list) -> bool:
        """
        Check if exception/constraint spans are retrieved.
        """
        for span in evidence_spans:
            text = span["text"].lower()
            if any(marker in text for marker in ["except", "unless", "not", "no", "cannot"]):
                return True
        return False
```

**Benchmark Script:**
```python
# evaluation/run_evaluation.py

evaluator = QAEvaluator()

# Test 1: Faithfulness
answer = "The deadline is 11th February 2026, 23:59"
evidence = [{"text": "Deadline: 11th February 2026, 23:59"}]
faith_score = evaluator.evaluate_faithfulness(answer, evidence)
print(f"Faithfulness: {faith_score:.2f}")

# Test 2: Robustness
queries = [
    "When is the assignment deadline?",
    "What is the due date for the assignment?",
    "Assignment ki deadline kab hai?"
]
robust_score = evaluator.evaluate_robustness(queries, expected_spans)
print(f"Robustness: {robust_score['avg_overlap']:.2f}")

# Test 3: Exception handling
query = "Can I get an extension?"
results = reasoner.hybrid_reasoning(query)
has_exception = evaluator.evaluate_exception_handling(query, results["span_results"])
print(f"Exception handling: {'✓' if has_exception else '✗'}")
```

---

## 🚀 Phase 5: Visualization (Optional - Week 5)

### Objective
Make reasoning interpretable with visual traces.

**File:** `visualization/graph_viz.py`

```python
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

def visualize_reasoning_trace(span_graph, reasoning_results):
    """Show which spans were retrieved and why."""
    
    # Highlight retrieved spans in graph
    retrieved_ids = set(reasoning_results["span_results"])
    
    # Create subgraph
    subgraph = span_graph.subgraph(retrieved_ids)
    
    # Plot with plotly
    pos = nx.spring_layout(subgraph)
    
    edge_trace = go.Scatter(...)  # edges
    node_trace = go.Scatter(...)  # nodes
    
    fig = go.Figure(data=[edge_trace, node_trace])
    st.plotly_chart(fig)
```

**Streamlit App:**
```python
# app.py

import streamlit as st
from parser.hybrid_reasoner import HybridReasoner

st.title("Hinglish Document QA System")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
query = st.text_input("Ask a question (English or Hinglish):")

if st.button("Get Answer"):
    # Process document
    # ... build graphs ...
    
    # Get results
    results = reasoner.hybrid_reasoning(query)
    
    # Show answer
    st.subheader("Answer:")
    st.write(answer)
    
    # Show evidence
    st.subheader("Evidence Spans:")
    for span_id in results["span_results"]:
        st.write(f"- {span_graph.nodes[span_id]['text']}")
    
    # Show graph
    st.subheader("Reasoning Trace:")
    visualize_reasoning_trace(span_graph, results)
```

Run:
```bash
streamlit run app.py
```

---

## 📊 Success Criteria (From Your Proposal)

| Criteria | Measurement | Target |
|----------|-------------|--------|
| **Faithfulness** | Evidence-answer alignment | > 0.85 |
| **Robustness** | Paraphrase retrieval consistency | > 0.75 |
| **Exception Handling** | Exception spans in results | > 90% |
| **Hinglish Support** | Code-mixed query accuracy | > 0.80 |
| **Interpretability** | Reasoning trace clarity | Qualitative ✓ |

---

## 🎯 4-Week Sprint Plan

### Week 1: Multilingual Foundation
- [ ] Replace with multilingual embeddings
- [ ] Create Hinglish test queries (10-15)
- [ ] Benchmark retrieval on English vs Hinglish
- [ ] Fix any bugs in span/KG extraction

### Week 2: Query Understanding
- [ ] Add query translation module
- [ ] Implement query expansion (synonyms)
- [ ] Test paraphrase robustness
- [ ] Optimize retrieval thresholds

### Week 3: Answer Generation
- [ ] Integrate LLM (GPT-4 or local Llama)
- [ ] Design evidence-grounded prompts
- [ ] Test Hinglish response generation
- [ ] Add answer quality checks

### Week 4: Evaluation & Polish
- [ ] Implement faithfulness metrics
- [ ] Run robustness tests
- [ ] Create comparison table (LLM-only vs Hybrid)
- [ ] Write final report
- [ ] Optional: Build Streamlit demo

---

## 🔗 Dependencies to Add

```bash
# requirements.txt additions

# Translation
googletrans==4.0.0-rc1
# OR
indic-trans

# Multilingual embeddings (already have sentence-transformers)

# LLM integration
openai
# OR for local models
transformers
torch
accelerate
bitsandbytes  # for quantization

# Visualization (optional)
streamlit
plotly
```

---

## 📝 Expected Deliverables for INLP Project

1. **Code Repository** ✅ (mostly done)
   - All modules implemented
   - Clean, documented code
   - README with setup instructions

2. **Technical Report** (4-6 pages)
   - Problem statement
   - Related work
   - System architecture
   - **Span graph + KG design**
   - **Hinglish query handling**
   - Evaluation results
   - Comparison with baselines

3. **Presentation** (10-15 slides)
   - Motivation
   - Key contribution: Externalized reasoning + Hinglish
   - System demo
   - Results and analysis

4. **Demo** (Optional but impressive)
   - Streamlit web app
   - Upload PDF, ask Hinglish questions
   - Show reasoning traces

---

## 🚨 Common Pitfalls to Avoid

1. **Don't over-engineer query translation**
   - Multilingual embeddings handle most cases
   - Simple translation API is fine

2. **Don't skip evaluation**
   - Your proposal emphasizes faithfulness
   - Need quantitative results

3. **Don't ignore exceptions**
   - Show span graph captures "no extension" correctly
   - This is a key advantage over flat retrieval

4. **Don't claim LLM-free if using GPT-4**
   - Be clear: LLM for generation only, not reasoning
   - Reasoning is pure graph traversal

---

## ✅ Final Checklist Before Submission

- [ ] System runs end-to-end (PDF → Hinglish query → Answer)
- [ ] At least 10 Hinglish test queries
- [ ] Evaluation metrics computed
- [ ] Comparison table (Sentence-DRG vs Span+KG vs Hybrid)
- [ ] README.md updated with Hinglish examples
- [ ] Code is clean and documented
- [ ] Technical report written
- [ ] Presentation slides ready
- [ ] Demo video recorded (optional)

---

Good luck! 🚀 You now have:
- ✅ Sentence-level DRG
- ✅ Span-level graph
- ✅ Knowledge graph
- ✅ Hybrid reasoning

Next: **Add Hinglish support** to make this uniquely yours! 🇮🇳
