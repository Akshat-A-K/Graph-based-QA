# Graph-Based QA

# Team: CTRL+ALT+DLT

Akshat 2025201005

Om 2025201008

Hardik 2025201046

Gaurav 2025201050

---

This project is for document question answering using graphs.
This README has information regarding the project setup and usage.

## Python Version

Use Python 3.14.

Check version:

```bash
python --version
```

## Files in This Project

- app.py: Streamlit app for PDF upload and question answering
- hotpot_dataset.py: run QA pipeline on Hotpot dataset
- hotpot_train_v1.1.json: Hotpot dataset file used by script
- setup_nltk.py: download NLTK data (stopwords, punkt)
- requirements.txt: Python packages
- parser/: all core logic (graphs, retrieval, reasoning)
- graphs/: output graph files (json, graphml, images)
- docs/: project docs/flow diagrams
- Presentation: Presentation.pdf
- Report: CTRL+ALT+DLT_Final.pdf

## One-Time Setup

1) Create virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install requirements

```bash
pip install -r requirements.txt
```

3) Setup NLTK (required)

```bash
python setup_nltk.py
```

4) Install spaCy models

```bash
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_sm
```

## Quick Start

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python setup_nltk.py
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_sm
streamlit run app.py
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
python3 setup_nltk.py
python3 -m spacy download en_core_web_trf
python3 -m spacy download en_core_web_sm
streamlit run app.py
```

## Run the App (app.py)

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually http://localhost:8501).

Linux alternate run:

```bash
python3 -m streamlit run app.py
```

## Run Hotpot Dataset Script

Default run:

```bash
python hotpot_dataset.py
```

Run specific number of questions:

```bash
python hotpot_dataset.py --num 100
```

Run all selected questions:

```bash
python hotpot_dataset.py --num 0
```

Run by level:

```bash
python hotpot_dataset.py --level easy --num 50
python hotpot_dataset.py --level medium --num 50
python hotpot_dataset.py --level hard --num 50
```

## Run LLM Evaluation Only

Run the LLM baselines on the same sampled questions used by the graph pipeline:

```bash
python -m evaluation.run_eval --mode llm --num 6 --seed 42
```

This evaluates the closed-book, open-book, and naive RAG baselines on 6
questions and writes the results to `results/llm_eval_results.json` and
`results/llm_eval_results.txt`.

If you want to run only the LLM evaluator module directly, you can also use:

```bash
python -m evaluation.llm.llm_eval --num 6 --seed 42
```

## Notes

- Keep hotpot_train_v1.1.json in project root for hotpot_dataset.py.
- If streamlit command is not found, use:

```bash
python -m streamlit run app.py
```

- Linux alternative:

```bash
python3 -m streamlit run app.py
```

## Presentation and Results

Team Name: CTRL+ALT+DLT

HotpotQA (with KG) summary:

- Questions evaluated: 500
- Answered: 500 (100.0%)
- Exact Match (EM): 0.3700 (37.0%)
- F1 Score: 0.4817 (48.2%)
- EM bridge / compare: 0.4257 / 0.3147
- F1 bridge / compare: 0.5139 / 0.4498
- Avg reasoning depth: 6.10
- Avg KG nodes/edges: 72 / 56
- Avg time/question: 54.52s
- Total pipeline time: 463.7 min (27820s)

Useful presentation assets:

- Flow diagram: docs/flow_diagram.svg
- Timeline diagram: docs/timeline_diagram.pdf
- Architecture diagram: docs/architecture_diagram.pdf
- DRG graph image: graphs/drg_graph.png
- Span graph image: graphs/span_graph.png
- Presentation file: Presentation.pdf
- Final report PDF (add this file): CTRL+ALT+DLT_Final.pdf
