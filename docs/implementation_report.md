# Graph-based-QA Implementation Report

This report documents the implementation that is actually present in the repository. It is based on the reviewed source code only, and it intentionally avoids external assumptions or benchmark claims that are not in the code.

## Scope

The repository implements a document question answering pipeline with three main layers:

1. Sentence-level reasoning over a Document Reasoning Graph.
2. Span-level reasoning over a fine-grained Span Graph.
3. Entity-level reasoning over a Knowledge Graph.

The Streamlit app is a user interface wrapper around the same core pipeline. The core logic lives in the `parser/` modules, while the top-level scripts provide different runtime modes for interactive QA, batch QA, HotpotQA evaluation, and dataset inspection.

## Reviewed Sources

| File                                                         | Role                                                                                                            |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| [app.py](../app.py)                                             | Streamlit interface for PDF upload, graph construction, question answering, evidence display, and graph export. |
| [run_custom_qa.py](../run_custom_qa.py)                         | Single-PDF CLI runner with canned question sets, graph export, and answer reporting.                            |
| [hotpot_dataset.py](../hotpot_dataset.py)                       | HotpotQA evaluation harness with level/type sampling, KG-augmented reasoning, and metric aggregation.           |
| [show_comparisons.py](../show_comparisons.py)                   | Utility for printing comparison-style HotpotQA questions.                                                       |
| [count_question_difficulty.py](../count_question_difficulty.py) | Utility for counting easy/medium/hard HotpotQA items.                                                           |
| [setup_nltk.py](../setup_nltk.py)                               | One-time NLTK data downloader.                                                                                  |
| [requirements.txt](../requirements.txt)                         | Runtime and optional dependency list.                                                                           |
| [parser/config.py](../parser/config.py)                         | Environment-based PDF extraction configuration.                                                                 |
| [parser/pdf_parser.py](../parser/pdf_parser.py)                 | PDF text, table, OCR, and section extraction.                                                                   |
| [parser/sentence_splitter.py](../parser/sentence_splitter.py)   | Sentence segmentation with spaCy and regex fallback.                                                            |
| [parser/section_utils.py](../parser/section_utils.py)           | Heuristic section header detection.                                                                             |
| [parser/drg_nodes.py](../parser/drg_nodes.py)                   | Converts pages into sentence nodes.                                                                             |
| [parser/drg_graph.py](../parser/drg_graph.py)                   | Sentence-level graph construction and export.                                                                   |
| [parser/span_extractor.py](../parser/span_extractor.py)         | Fine-grained span extraction from sentences.                                                                    |
| [parser/span_graph.py](../parser/span_graph.py)                 | Span-level graph construction and export.                                                                       |
| [parser/knowledge_graph.py](../parser/knowledge_graph.py)       | Triple-based Knowledge Graph construction and query APIs.                                                       |
| [parser/question_processor.py](../parser/question_processor.py) | Question type classification and decomposition.                                                                 |
| [parser/enhanced_reasoner.py](../parser/enhanced_reasoner.py)   | Main retrieval and reasoning orchestrator.                                                                      |
| [parser/advanced_retrieval.py](../parser/advanced_retrieval.py) | BM25, normalization, graph centrality, hybrid scoring, edge weighting.                                          |
| [parser/answer_selector.py](../parser/answer_selector.py)       | Final answer span selection and extractive QA pass.                                                             |
| [parser/evaluator.py](../parser/evaluator.py)                   | Evaluation metrics used by the runtime scripts.                                                                 |
| [parser/model_cache.py](../parser/model_cache.py)               | Central cache for sentence-transformer, QA, and NER pipelines.                                                  |
| [parser/comparison_utils.py](../parser/comparison_utils.py)     | Hotpot comparison-question heuristics.                                                                          |
| [parser/__init__.py](../parser/__init__.py)               | Empty package initializer.                                                                                      |

## End-to-End Flow

The system flow is:

1. A PDF is uploaded or provided on the command line.
2. The PDF is parsed into page text, blocks, optional tables, and optional OCR text.
3. Pages are converted into sentence nodes.
4. Sentence nodes are expanded into span nodes.
5. A sentence graph, span graph, and knowledge graph are built from the extracted text.
6. The reasoning engine ranks candidate spans using semantic similarity, lexical overlap, graph centrality, traversal, query expansion, and KG evidence.
7. The answer selector optionally runs an extractive QA model to refine the final answer text.
8. Evidence spans, reasoning chains, confidence, and metrics are returned to the UI or written to disk.

The repository also contains two source diagrams that describe the same overall flow:

| Diagram                                                              | Purpose                                        |
| -------------------------------------------------------------------- | ---------------------------------------------- |
| [docs/codebase_flowchart.mmd](codebase_flowchart.mmd)                   | Mermaid version of the PDF -> answer pipeline. |
| [docs/codebase_flowchart_concise.puml](codebase_flowchart_concise.puml) | PlantUML version of the same method flow.      |

Additional rendered assets are present in `docs/`, including the SVG flow diagram and PDF diagrams.

## Runtime Entry Points

### `app.py`

The Streamlit app is the interactive front end. It does not train anything; it builds the graphs for the uploaded PDF in memory and lets the user ask questions.

Key behavior:

1. The user uploads a PDF and clicks the Process PDF button.
2. The app writes the bytes to a temporary PDF file and removes the file after processing.
3. It extracts the document with `extract_document_with_tables`.
4. It builds sentence nodes, a Document Reasoning Graph, a Span Graph, and a Knowledge Graph.
5. It initializes the reasoner with cross-encoder reranking disabled for the UI path.
6. On each question, it runs `enhanced_reasoning(question, k=5)` and then `select_answer(...)`.
7. It shows the answer, confidence, evidence spans, reasoning chains, retrieval breakdown, and document statistics.

The app uses the embedding model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` so that the UI path stays aligned with multilingual queries. The visible copy explicitly advertises English, Hindi, and Hinglish support.

### `run_custom_qa.py`

This is the single-PDF batch runner. It is designed for running a fixed set of questions against a single document and writing a JSON and TXT report beside the input PDF.

CLI behavior:

| Argument or flag     | Meaning                                                                                        |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| positional PDF path  | Input PDF to analyze. If omitted, the script falls back to a hardcoded local development path. |
| `--enable-ocr`     | Sets `PDF_ENABLE_OCR=true` before importing the parser.                                      |
| `--disable-tables` | Sets `PDF_ENABLE_TABLES=false` before importing the parser.                                  |

Question routing is filename-based:

| Filename trigger                                                                                                  | Question set                                                                                                                               |
| ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| contains `a2`                                                                                                   | Assignment-style questions about deadlines, embeddings, POS tagging, splits, and report requirements.                                      |
| contains `test`                                                                                                 | Abdul Kalam biography questions.                                                                                                           |
| contains any of `rsearch`, `pepar`, `research`, `paper`, `eurocall`, `kruk`, `mobile`, `autonomy` | Research-paper questions covering factual, definitional, methodological, process, inferential, comparative, evaluative, and causal themes. |
| otherwise                                                                                                         | Generic document questions.                                                                                                                |

The script exports:

| Artifact                         | Location                             |
| -------------------------------- | ------------------------------------ |
| JSON QA report                   | `*_qa_output.json` next to the PDF |
| Text QA report                   | `*_qa_output.txt` next to the PDF  |
| Span graph image and JSON        | `graphs/`                          |
| Knowledge graph JSON and GraphML | `graphs/`                          |
| DRG image                        | `graphs/`                          |

It also performs explicit garbage collection during the run and clears CUDA memory if available.

### `hotpot_dataset.py`

This is the HotpotQA benchmark runner. It converts each HotpotQA context into pseudo-pages, builds the graphs, runs the reasoner, and compares predictions against the gold answer.

CLI arguments:

| Argument             | Default                    | Meaning                                                           |
| -------------------- | -------------------------- | ----------------------------------------------------------------- |
| `--dataset`        | `hotpot_train_v1.1.json` | Input HotpotQA file.                                              |
| `--num`            | `50`                     | Number of questions to sample.`0` means all selected questions. |
| `--seed`           | `42`                     | Random seed for deterministic sampling.                           |
| `--embed-model`    | `BAAI/bge-large-en-v1.5` | Sentence embedding model used for the run.                        |
| `--level`          | `all`                    | `easy`, `medium`, `hard`, or `all`.                       |
| `--drg-threshold`  | `0.75`                   | Semantic threshold for the sentence graph.                        |
| `--span-threshold` | `0.70`                   | Semantic threshold for the span graph.                            |
| `--kg-model`       | `en_core_web_trf`        | spaCy model for the knowledge graph.                              |
| `--kg-hops`        | `2`                      | Maximum KG hops when extracting bridge evidence.                  |

Sampling strategy when `--level all`:

1. The dataset is split into easy, medium, and hard groups.
2. Each level is further split into bridge and comparison pools.
3. The requested sample size is split across levels, with any remainder assigned to earlier levels.
4. Each level receives a bridge/comparison split, with bridge questions getting roughly half the slots.

The script writes `hotpot_all_qa_output.json` and `hotpot_all_qa_output.txt` into the dataset directory.

### Utility scripts

`show_comparisons.py` loads HotpotQA, filters `type == comparison`, shuffles with a seed, and prints the selected questions. The `--show-context` flag is defined, but the context-printing block is commented out in the current code.

`count_question_difficulty.py` simply counts the number of easy, medium, and hard questions in a HotpotQA JSON file.

`setup_nltk.py` downloads the `stopwords` and `punkt` corpora and confirms that the English stopword list is available.

## PDF Ingestion and Text Normalization

### Configuration

`parser/config.py` reads three environment variables:

| Variable              | Default   | Effect                                                                                          |
| --------------------- | --------- | ----------------------------------------------------------------------------------------------- |
| `PDF_ENABLE_OCR`    | `false` | Enables OCR fallback for pages with no extracted text.                                          |
| `PDF_ENABLE_TABLES` | `true`  | Enables table extraction when `pdfplumber` is available.                                      |
| `PDF_OUTPUT_DIR`    | `.`     | Defines an output directory value, but the current runtime flow does not consume it downstream. |

The parser computes two effective flags at import time:

| Internal flag      | Definition                                           |
| ------------------ | ---------------------------------------------------- |
| `_ENABLE_OCR`    | `config.ENABLE_OCR and pytesseract availability`   |
| `_ENABLE_TABLES` | `config.ENABLE_TABLES and pdfplumber availability` |

Because these flags are resolved during import, `run_custom_qa.py` sets the environment variables before importing the parser modules.

### `parser/pdf_parser.py`

The PDF parser uses PyMuPDF (`fitz`) as the main extractor and optionally `pdfplumber` and OCR support.

The cleaning pipeline is:

1. Fix null-interleaved UTF-16 text runs.
2. Collapse spaced-character artifacts such as split currency or character-by-character runs.
3. Remove hyphenated line breaks.
4. Normalize line breaks and repeated whitespace.

Important text fixes:

| Helper                | Behavior                                                                                                                                                         |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_fix_null_encoded` | Removes null bytes in text like `\x00e\x00x\x00a\x00m\x00p\x00l\x00e`. It also maps the superscript-1 artifact back to the rupee sign when followed by digits. |
| `_fix_spaced_chars` | Collapses text that has one character per token and repairs separated currency symbols.                                                                          |
| `_clean_text`       | Applies the null-byte fix, spaced-character fix, hyphenation cleanup, and whitespace normalization.                                                              |

`extract_pages` returns a list of page dictionaries with the following keys:

| Key                  | Meaning                                                    |
| -------------------- | ---------------------------------------------------------- |
| `page`             | 1-based page number.                                       |
| `text`             | Cleaned plain text for the page.                           |
| `blocks`           | Text blocks with font-size metadata and heading hints.     |
| `median_font_size` | Median font size on the page, used for heading heuristics. |
| `metadata`         | Document metadata from PyMuPDF.                            |

Each block records `text`, `max_font_size`, `is_bold`, and a derived `is_heading` flag. A block is treated as a heading when its max font size is at least two points above the page median or when the block is bold.

OCR fallback is used only when the page has no text and `_ENABLE_OCR` is true.

Table extraction works in two ways:

1. If `pdfplumber` is available, `extract_tables()` is used page by page.
2. If it is not available, the parser falls back to a heuristic that groups consecutive lines with similar column counts after splitting on multiple spaces, pipes, tabs, or commas.

`extract_document` returns a document dictionary with `metadata`, `pages`, and `sections`. Section detection is page-heading based and is intentionally heuristic.

Important implementation note: `extract_document_with_tables` accepts `enable_tables` and `enable_ocr`, but the current `enable_ocr` branch simply re-calls `extract_document` and does not pass a per-call OCR override. In practice, OCR remains governed by the environment-based global flag.

### Section heuristics

`parser/section_utils.py` classifies a line as a section title when it matches one of these patterns:

1. Numbered heading like `1 Introduction` or `2.3 Methods`.
2. Title Case with eight words or fewer.
3. ALL CAPS with two to nine words.

Single-word lines and very short lines are ignored.

## Sentence and Span Construction

### Sentence nodes

`parser/drg_nodes.py` converts pages into sentence nodes.

Behavior:

1. It scans each page for heading lines and updates `current_section` when a heading is found.
2. It splits the page text into sentences using `split_into_sentences`.
3. It assigns a globally increasing `node_id` to each sentence.

Each sentence node has:

| Field          | Meaning                                        |
| -------------- | ---------------------------------------------- |
| `node_id`    | Global sentence identifier.                    |
| `text`       | Sentence text.                                 |
| `page`       | Source page number.                            |
| `section`    | Current inferred section title, or `GLOBAL`. |
| `sent_index` | Sentence index within the page.                |

### Sentence splitting

`parser/sentence_splitter.py` first tries spaCy sentence segmentation with `en_core_web_sm`. If spaCy is unavailable or fails, it falls back to regex splitting on sentence-ending punctuation, including the Hindi danda characters (U+0964 and U+0965).

The splitter removes sentences shorter than or equal to three characters.

### Span extraction

`parser/span_extractor.py` creates finer-grained spans from each sentence. The default NER model is `dslim/bert-base-NER`, loaded through the model cache.

Span extraction rules:

1. The full sentence is always included as a span.
2. NER spans are extracted when the NER pipeline is available.
3. Dependency-based spans are extracted through spaCy noun chunks, numeric noun phrases, and verb clauses.
4. Clauses are split only for sentences longer than 150 characters, using `;` or `:` followed by a capital letter. Clauses shorter than 30 characters are ignored.

The span extractor uses a de-duplication set of character ranges so overlapping extractions are suppressed.

Span fields:

| Field                         | Meaning                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------- |
| `span_id`                   | Span identifier within the extracted span list.                                 |
| `text`                      | Span text.                                                                      |
| `start_char` / `end_char` | Span character offsets relative to the sentence.                                |
| `span_type`                 | `sentence`, `ner:<label>`, `noun_phrase`, `verb_clause`, or `clause`. |
| `page`                      | Source page number.                                                             |
| `section`                   | Inferred section title.                                                         |
| `sentence_id`               | Parent sentence node id.                                                        |
| `entities`                  | Extracted entity strings attached to the span.                                  |

Important note: the current main entrypoints use `SpanExtractor.extract_spans_from_nodes(...)` directly. They do not call the full `SpanGraph.build_graph(...)` helper, so some `SpanGraph` capabilities are available in the class but are not activated by the main scripts.

## Graph Layer 1: Document Reasoning Graph

`parser/drg_graph.py` implements a directed sentence graph over the sentence nodes.

### Node attributes

Each sentence node stores:

| Attribute       | Meaning                                          |
| --------------- | ------------------------------------------------ |
| `text`        | Sentence text.                                   |
| `page`        | Source page number.                              |
| `section`     | Inferred section title.                          |
| `sent_index`  | Sentence index within the page.                  |
| `embedding`   | Sentence-transformer embedding.                  |
| `importance`  | TF-IDF importance score.                         |
| `length`      | Token count.                                     |
| `entities`    | Pattern-based or model-based entities.           |
| `pagerank`    | PageRank score after graph metrics are computed. |
| `degree`      | Node degree after graph metrics are computed.    |
| `kg_entities` | KG entities matched to the sentence, if any.     |

### Embeddings and importance

The DRG uses a sentence-transformer model loaded through the cache. If model loading fails, the code falls back to zero embeddings.

The TF-IDF importance score is computed with `TfidfVectorizer(max_features=100, stop_words='english')`.

If TF-IDF fails, importance falls back to 1.0 for every node.

### Structural edges

The DRG class supports these edge types:

| Edge type     | Condition                                                | Weight           |
| ------------- | -------------------------------------------------------- | ---------------- |
| `adjacent`  | Consecutive sentences on the same page.                  | `1.0`          |
| `context`   | Backward link from the next sentence to the current one. | `0.8`          |
| `page`      | Same page, non-adjacent.                                 | `0.5`          |
| `section`   | Same non-global section.                                 | `0.6`          |
| `proximity` | Same page and within three sentence positions.           | `1 / distance` |

### Semantic edges

The semantic edge builder computes the full cosine similarity matrix between sentence embeddings.

The thresholding rule is:

$$
\text{dynamic\_threshold} = \max(\text{threshold}, P_{85}(\text{all pairwise similarities}))
$$

The default threshold is `0.75`. Bidirectional semantic edges are added for pairs above the dynamic threshold.

### Entity extraction and KG alignment

The DRG extracts entities in two ways:

1. Optional NER using `dslim/bert-base-NER` when `enable_model_ner=True`.
2. Pattern fallback for proper nouns, 4-digit years, dates, percentages, and score-like expressions.

Pattern-based entity labels include `PERSON/ORG`, `YEAR`, `DATE`, `PERCENTAGE`, and `SCORE`.

`add_kg_edges` matches KG entities against sentence text with word-boundary regular expressions. If a sentence matches a KG entity, the sentence metadata is augmented with `kg_entities`. Sentences that share KG entities are connected with `kg_overlap` edges of weight `0.9`.

Entity coreference edges inside the DRG use weight `0.85`.

### Graph metrics and export

The DRG computes:

1. PageRank with NetworkX using edge weights.
2. Node degree.

It also exports a PNG graph image for visualization.

### Runtime usage note

The class supports structural edges, but the batch scripts do not all use the same edge set:

| Runtime path          | DRG edges actually added                                                             |
| --------------------- | ------------------------------------------------------------------------------------ |
| Streamlit app         | Structural + semantic + KG alignment.                                                |
| `run_custom_qa.py`  | Semantic + KG alignment.                                                             |
| `hotpot_dataset.py` | Semantic only for the DRG itself; KG is built separately and passed to the reasoner. |

## Graph Layer 2: Span Graph

`parser/span_graph.py` is the span-level graph builder. It is more detailed than the DRG and supports entity nodes, discourse edges, and entity overlap edges.

### Default model and importance

The class default embedding model is `sentence-transformers/multi-qa-mpnet-base-dot-v1`, but the main scripts usually override it with their own run-specific embedding model.

Span importance is the word count of the span text.

### Span graph node types

| Node type    | Meaning                                                                                    |
| ------------ | ------------------------------------------------------------------------------------------ |
| Span nodes   | The actual extracted spans from `SpanExtractor`.                                         |
| Entity nodes | Synthetic nodes created from span entities using the format `entity::<normalized_text>`. |

Entity nodes use `page=-1`, `section=GLOBAL`, and `sentence_id=-1`.

### Span graph edges

| Edge type                                                                                                            | Condition                                                                                          | Weight           |
| -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ---------------- |
| `same_sentence`                                                                                                    | Two spans come from the same sentence.                                                             | `1.0`          |
| `sequential`                                                                                                       | Two spans are sequential in node order.                                                            | `0.9`          |
| `backward`                                                                                                         | Reverse link for sequential spans.                                                                 | `0.7`          |
| `semantic`                                                                                                         | Cosine similarity above threshold.                                                                 | Similarity score |
| `condition`, `exception`, `temporal`, `negation`, `requirement`, `causation`, `contrast`, `addition` | Discourse marker detected in the node text and the other span is in the same or adjacent sentence. | `0.85`         |
| `entity`                                                                                                           | Two spans share an extracted entity.                                                               | `0.8`          |
| `entity_relation`                                                                                                  | Two entity nodes appear in the same span.                                                          | `0.9`          |
| `mentions` / `mentioned_in`                                                                                      | Span to entity-node links.                                                                         | `1.0`          |

### Semantic thresholding

The span graph uses the same adaptive threshold pattern as the DRG, but with a higher percentile cutoff in the class implementation:

$$
\text{dynamic\_threshold} = \max(\text{threshold}, P_{80}(\text{all pairwise similarities}))
$$

The class default threshold is `0.75`, and the main scripts pass `0.70` for the CLI and HotpotQA flows.

The class also limits semantic fan-out to `max_neighbors=3` per node.

### Discourse markers

The discourse categories and keywords are:

| Category    | Keywords                                           |
| ----------- | -------------------------------------------------- |
| condition   | if, unless, provided, only if, in case, when       |
| exception   | except, excluding, but not, other than, apart from |
| temporal    | before, after, until, by, deadline, during, while  |
| negation    | not, no, never, cannot, won't, shouldn't           |
| requirement | must, shall, required, need to, have to, should    |
| causation   | because, since, therefore, thus, hence, so         |
| contrast    | but, however, although, while, whereas, yet        |
| addition    | and, also, moreover, furthermore, additionally     |

### Full build helper versus actual runtime usage

`SpanGraph.build_graph(...)` would build the full feature set, including entity nodes and entity relations. However, the main scripts do not all use that helper.

| Runtime path          | Span graph edge set actually used      |
| --------------------- | -------------------------------------- |
| Streamlit app         | Structural + semantic + discourse.     |
| `run_custom_qa.py`  | Semantic + discourse + entity overlap. |
| `hotpot_dataset.py` | Semantic + discourse + entity overlap. |

This means entity nodes and entity-entity links exist in the class, but they are not activated by the current app and batch scripts.

### Export functions

The span graph can be exported as JSON, GraphML, and a PNG image.

## Graph Layer 3: Knowledge Graph

`parser/knowledge_graph.py` builds a directed graph from triples extracted with spaCy dependency parsing.

### Model loading order

The knowledge graph tries spaCy models in this order:

1. The requested model, default `en_core_web_trf`.
2. `en_core_web_lg`.
3. `en_core_web_sm`.

If spaCy itself is unavailable, KG extraction is disabled.

### Triple extraction

The KG extracts triples from each sentence using these patterns:

1. Verbal predicates with subjects and direct objects.
2. Passive voice, which is flipped for readability.
3. Prepositional chains such as verb + preposition + object.
4. Copular `be` constructions, which become `is_a` edges.
5. Noun-chunk fallback when no verb triples are found, connecting consecutive named entities with `related_to` so the graph remains connected.

Triple types in the actual code path are `relation` and `is_a`, with fallback `related_to` edges.

### Node and edge attributes

| Attribute     | Meaning                                          |
| ------------- | ------------------------------------------------ |
| `text`      | Normalized entity text.                          |
| `label`     | NER label or `entity`.                         |
| `frequency` | Count of how many times the entity was observed. |
| `pagerank`  | PageRank score.                                  |
| `community` | Community id from modularity clustering.         |

Edge attributes include `relation`, `type`, and `weight`. If the same edge is discovered again, its weight is incremented by `1.0`.

### Analytics

The KG computes:

1. PageRank with `alpha=0.85` and `max_iter=200`.
2. Greedy modularity communities on the undirected version, when the community module is available.

The `get_stats()` method returns:

| Field             | Meaning                         |
| ----------------- | ------------------------------- |
| `nodes`         | Number of KG nodes.             |
| `edges`         | Number of KG edges.             |
| `communities`   | Number of detected communities. |
| `entity_labels` | Label histogram.                |
| `density`       | NetworkX density.               |
| `is_connected`  | Weak connectivity flag.         |

### Query APIs

The KG exposes the following methods:

| Method                                       | Purpose                                                        |
| -------------------------------------------- | -------------------------------------------------------------- |
| `query_entity(entity, max_hops=2)`         | Returns triples reachable within the hop limit from an entity. |
| `shortest_path_evidence(entity1, entity2)` | Returns relation-labeled path evidence between two entities.   |
| `entity_bridge(entity1, entity2)`          | Returns the intermediate bridge nodes on the shortest path.    |
| `top_entities(n=10)`                       | Returns the top entities by PageRank.                          |
| `entities_by_label(label)`                 | Returns all entities with a specific NER label.                |

The `build_graph` method accepts either a list of strings or a list of sentence-node dictionaries. This is why the entrypoints can pass either `sentence_texts` or `sentence_nodes`.

## Question Processing

`parser/question_processor.py` handles question classification and decomposition.

### Default transformer models

| Component              | Default model                |
| ---------------------- | ---------------------------- |
| Question classifier    | `facebook/bart-large-mnli` |
| Sub-question generator | `google/flan-t5-base`      |

If the transformers pipelines fail to load, the module falls back to rule-based processing.

### Question type inventory

The internal type list is:

`factual`, `definitional`, `methodological`, `comparative`, `causal`, `process`, `temporal`, `bridge`

### Rule-based fallback

The fallback classifier uses keyword heuristics:

| Pattern                                                                               | Returned type |
| ------------------------------------------------------------------------------------- | ------------- |
| compare, difference, versus, vs, older, younger, taller, which is, are both, or, both | comparative   |
| why, because, reason                                                                  | causal        |
| when, date, year, deadline                                                            | temporal      |
| how, process, step, method                                                            | process       |
| what is, define, meaning                                                              | definitional  |
| otherwise                                                                             | factual       |

Important note: the rule-based fallback does not explicitly emit `bridge`; the `bridge` type depends on the zero-shot classifier or on downstream HotpotQA-specific logic.

### Sub-question generation

Only complex types are decomposed: `comparative`, `bridge`, `causal`, and `process`.

The decomposition strategy is:

1. If the generator pipeline is available, ask it to decompose the question into 2-3 simpler sub-questions.
2. Otherwise use pattern-based splitting.

Pattern-based rules include:

| Pattern                                   | Output                                                    |
| ----------------------------------------- | --------------------------------------------------------- |
| `Are X and Y both Z?`                   | `Is X Z?`, `Is Y Z?`                                  |
| `Which who/what/what ... A and B`       | Two separate sub-questions with A and B.                  |
| `Who/Which/What is X or Y?`             | Two comparative follow-up questions.                      |
| `Does A have property that B does not?` | Separate property questions for each entity.              |
| `the X of Y`                            | Bridge-style fallback with a definitional sub-question.   |
| plain `and` split                       | Split into two questions when both sides are long enough. |

The `process()` method returns a dictionary with `original_question`, `type`, `sub_questions`, and `is_complex`.

## Retrieval, Ranking, and Reasoning

`parser/enhanced_reasoner.py` is the main orchestration layer. It combines BM25, semantic similarity, graph centrality, graph traversal, query expansion, KG-guided retrieval, optional cross-encoder reranking, and evidence-chain construction.

### Initialization

The reasoner stores the sentence graph, span graph, and optional KG graph. It then:

1. Loads a sentence-transformer query embedding model.
2. Initializes BM25 indices for spans and sentences.
3. Computes PageRank and betweenness centrality for the span graph.
4. Computes PageRank centrality for the sentence graph.
5. Optionally loads a cross-encoder reranker.

The default cross-encoder is `cross-encoder/ms-marco-MiniLM-L-6-v2`.

The app disables cross-encoder reranking for the UI path, while the batch scripts keep the default enabled behavior unless model loading fails.

### Centrality signals

Span centrality is the mean of PageRank and betweenness:

$$
\text{span\_centrality}(v) = \frac{\text{PageRank}(v) + \text{Betweenness}(v)}{2}
$$

Sentence centrality is just PageRank.

### BM25 helper

`parser/advanced_retrieval.py` implements BM25 with the standard formula:

$$
\mathrm{IDF}(t) = \log\left(\frac{N - df_t + 0.5}{df_t + 0.5} + 1\right)
$$

$$
\mathrm{BM25}(q, d) = \sum_{t \in q} \mathrm{IDF}(t) \cdot
\frac{f(t,d)(k_1 + 1)}{f(t,d) + k_1\left(1 - b + b\frac{|d|}{\mathrm{avgdl}}\right)}
$$

The implemented BM25 parameters are `k1=1.5` and `b=0.75`.

### Hybrid scoring helper

`HybridScorer` normalizes each score list with min-max scaling, then applies weighted sum combination. The default weights are:

| Signal     | Weight  |
| ---------- | ------- |
| Semantic   | `0.5` |
| Lexical    | `0.3` |
| Centrality | `0.2` |

### Sentence scoring

The reasoner computes a sentence-level relevance score before final candidate scoring:

$$
\text{sentence\_score} = 0.45\,\text{semantic} + 0.25\,\text{overlap} + 0.20\,\text{BM25} + 0.10\,\text{centrality}
$$

The overlap term is the token overlap ratio between the query and the sentence.

### Span retrieval stages

The reasoner uses four retrieval stages:

1. `enhanced_span_retrieval` for hybrid semantic/lexical/centrality ranking.
2. `enhanced_span_traversal` for graph-neighborhood expansion.
3. `retrieval_with_expansion` for query-variant expansion and reciprocal-rank merging.
4. `kg_guided_retrieval` for entity-driven retrieval from the KG.

#### 1. Hybrid span retrieval

For each span, the method computes:

1. Semantic similarity between query and span embeddings.
2. A small overlap bonus up to `0.15`.
3. BM25 lexical score.
4. Span centrality.

The semantic score receives a light lexical bonus:

$$
\text{semantic\_score} = \cos(q, s) + \min(0.15, 0.03 \cdot \text{overlap}) + 0.03\,\mathbf{1}[\text{query tokens intersect sentence tokens}]
$$

The combined ranking uses the `HybridScorer` over normalized semantic, lexical, and centrality signals.

#### 2. Graph traversal

Traversal starts from the top hybrid seeds, expands through up to five neighbors per hop, and prioritizes discourse-related edges such as condition, exception, temporal, and same_sentence.

The scoring for traversed nodes is:

$$
\text{traversal\_score} = \cos(q, s) + 0.2\,\text{centrality} + 0.1\,\text{discourse bonus} + \min(0.2, 0.04 \cdot \text{token overlap})
$$

The traversal depth is `2` in the main reasoning call.

#### 3. Query expansion

Query expansion is heuristic token augmentation, not generative expansion. It:

1. Finds the top 3 semantic spans.
2. Takes up to 4 new tokens from each span that are not already in the query.
3. Creates text variations and deduplicates them with normalized text matching.
4. Merges retrieval results by reciprocal rank:

$$
\text{score}(s) = \sum_i \frac{1}{\text{rank}_i(s) + 1}
$$

#### 4. KG-guided retrieval

The reasoner searches the KG for entities mentioned in the query, expands them to one-hop neighbors, and then retrieves spans that mention any of those expanded entities.

### Final scoring inside `_run_reasoning_core`

The final candidate score is a weighted combination of semantic, lexical, overlap, sentence-level, centrality, retrieval-vote, and KG signals:

$$
\begin{aligned}
\text{final\_score} =
&\ 0.28\,\text{semantic} + 0.18\,\text{BM25} + 0.12\,\text{overlap} + 0.10\,\text{sentence\_bonus} \\
&+ 0.08\,\text{centrality} + 0.06\,\mathbf{1}[\text{hybrid}] + 0.04\,\mathbf{1}[\text{traversal}] \\
&+ 0.03\,\mathbf{1}[\text{expansion}] + 0.05\,\mathbf{1}[\text{KG}] + 0.06\,\text{kg\_bonus}
\end{aligned}
$$

The KG bonus adds `0.05` for each KG entity matched in the span, capped at `0.15`.

### Candidate filtering and diversity

After ranking, the reasoner:

1. Filters candidates with a minimum semantic threshold and minimal overlap rules.
2. Applies MMR diversity reranking.
3. Adds a small number of evidence-chain nodes if they are highly relevant.
4. Returns the final span list and supporting metadata.

The MMR formula is:

$$
\mathrm{MMR}(s) = \lambda\,\mathrm{sim}(s, q) - (1-\lambda)\max_{s_j \in S} \mathrm{sim}(s, s_j)
$$

The main reasoning path uses `lambda = 0.65`.

### Complex-question aggregation

If the question is classified as complex, the reasoner decomposes it into sub-questions, runs the core reasoning on each sub-question, and aggregates the results.

The aggregated result schema contains:

| Key                     | Meaning                                            |
| ----------------------- | -------------------------------------------------- |
| `final_spans`         | Final merged span ids.                             |
| `span_scores`         | Combined span scores across sub-questions.         |
| `kg_entities`         | Union of KG entities touched by the sub-questions. |
| `evidence_chains`     | Top unique chains.                                 |
| `is_aggregated`       | True when decomposition was used.                  |
| `sub_questions_count` | Number of sub-questions executed.                  |

### Evidence chains

`build_evidence_chains` creates two kinds of chain evidence:

1. DRG paths between the sentence ids of top spans. Paths are limited to lengths between 2 and 4.
2. KG bridge paths between query entities and KG entities found in the seed sentences. Paths are limited to lengths between 2 and 3.

DRG chain score is the average PageRank of the nodes on the path. KG bridge chain score is `0.5 / path_length`.

### Return schema from `enhanced_reasoning`

The non-aggregated result dictionary includes:

| Key                   | Meaning                                            |
| --------------------- | -------------------------------------------------- |
| `final_spans`       | Final span ids to consider for answer selection.   |
| `hybrid_results`    | Top hybrid retrieval spans.                        |
| `traversal_results` | Top traversal spans.                               |
| `expansion_results` | Top expanded-query spans.                          |
| `kg_results`        | Top KG-guided spans.                               |
| `kg_entities`       | KG entities encountered in evidence chains.        |
| `kg_spans`          | Span ids appearing in evidence chains.             |
| `evidence_chains`   | Chain objects with text, score, and node sequence. |
| `span_scores`       | Final span score map.                              |

## Final Answer Selection

`parser/answer_selector.py` takes the reasoner output and chooses the final answer text.

### Candidate collection

The candidate pool is built from:

1. `final_spans`
2. `hybrid_results`
3. `traversal_results`
4. `expansion_results`
5. `kg_spans`
6. `kg_results`

The selector deduplicates candidates and ranks up to the first 40 span ids.

### Ranking formula

For each candidate span, the selector starts from the reasoner score and adjusts it as follows:

1. Add `0.08` if the span type is `sentence`.
2. Add `0.2 * overlap_ratio`, where `overlap_ratio = overlap / max(1, len(query_content))`.
3. Add `0.08` if the overlap count is at least 2.
4. Subtract `0.12` if the overlap count is 0.
5. Subtract `0.08` if the span is extremely long, defined as longer than `1.6 * max_length`.
6. Subtract `0.0003 * min(len(text), 600)` as a light length penalty.

The selector keeps at most two spans per sentence id during ranking, unless that leaves fewer than three candidates.

### Fallback and truncation rules

If the best span has zero lexical overlap, the selector scans the whole span graph for any overlap-bearing span and uses the best fallback.

If the winning span is not a sentence span, the selector prefers the parent sentence when it exists in the reasoner sentence graph and is short enough.

The final answer text is truncated to `max_length`. If possible, it is cut at the last period or Hindi danda before the cutoff; otherwise it is shortened on a word boundary with an ellipsis.

### Extractive QA refinement

The selector then runs an extractive question-answering model over one or more contexts.

Pipeline behavior:

1. Contexts include the top ranked passage, up to nine runner-up passages, and the parent sentence when available.
2. Contexts may be reordered by BM25 if the BM25 top context is meaningfully better than the current first context.
3. Pass 1 runs QA on each context individually.
4. Pass 2 runs QA on a concatenated context when the best score is still low.
5. Pass 3 runs BM25 over the full sentence inventory when the answer is still uncertain.

The extractive QA call uses:

| Setting            | Value                               |
| ------------------ | ----------------------------------- |
| Model              | `deepset/deberta-v3-large-squad2` |
| `max_answer_len` | `200`                             |
| `top_k`          | `3`                               |

The code computes a combined confidence score as:

$$
\text{confidence} = 0.4\,\text{normalized retrieval score} + 0.6\,\text{QA score}
$$

If the QA model does not produce a score, the confidence falls back to the normalized retrieval score.

The selector returns:

1. Final answer text.
2. Up to three evidence texts.
3. Combined confidence.

## Evaluation Methodology

`parser/evaluator.py` defines the core runtime metrics used by the scripts.

### Text normalization

`QAEvaluator.normalize_text` lowercases text, removes punctuation, strips articles, and collapses whitespace.

### Exact match

Exact match returns 1.0 when the normalized prediction equals the normalized ground truth, and 0.0 otherwise.

### Precision, recall, and F1

The F1 computation is token overlap based:

$$
P = \frac{|pred \cap gt|}{|pred|}, \quad R = \frac{|pred \cap gt|}{|gt|}, \quad F_1 = \frac{2PR}{P+R}
$$

When there is no token overlap, the evaluator applies two fallback rules:

1. Substring containment can yield partial credit with precision and recall of 0.6.
2. Numeric answers are treated as correct when the first numeric tokens differ by at most 1 percent relative tolerance.

### Substring match

Substring match returns 1.0 when either normalized string contains the other.

### Evidence recall@k

The evidence recall metric checks whether any of the top-k evidence spans contains the ground truth or is contained by it after normalization.

### Reasoning depth

Reasoning depth is the number of unique nodes visited across traversal results and expansion results.

### HotpotQA evaluation helpers

`hotpot_dataset.py` also defines its own normalization and exact/F1 computation helpers for the gold-answer benchmark path. These follow the HotpotQA style of lowercasing, punctuation removal, article removal, and whitespace normalization.

### Script-level metric usage

| Script                | Primary evaluation target                                                                                  |
| --------------------- | ---------------------------------------------------------------------------------------------------------- |
| `app.py`            | Interactive display only; no written metric report.                                                        |
| `run_custom_qa.py`  | Answer quality versus top evidence, recall@5, reasoning depth, runtime, confidence, retrieval counts.      |
| `hotpot_dataset.py` | Gold-answer EM/F1, internal evidence metrics, type-wise breakdown, reasoning depth, KG stats, and runtime. |

## HotpotQA Comparison and Bridge Logic

`parser/comparison_utils.py` is used by `hotpot_dataset.py` for comparison questions.

### Comparison types

`classify_comparison_type` returns one of:

`difference`, `common_property`, `boolean`, `comparative`, `select_one`

The heuristics check for patterns such as:

1. Difference language like `different` or `how is`.
2. Common-property phrases such as `in common`, `shared`, `what type of`, `what kind of`.
3. Boolean and both/or patterns.
4. Comparative ranking words like `more`, `less`, `larger`, `smaller`, `older`, `earlier`, `first`, `recent`.

### Entity extraction for comparisons

`extract_comparison_entities` tries several patterns in order:

1. `between X and Y`
2. Comma-separated `X or Y`
3. Semicolon-separated `X or Y`
4. `both X and Y`
5. `X and Y, ...`
6. `Are/Is X and Y both ...`
7. General `X and Y` before a verb or copula
8. Fallback `X or Y`

### Hotpot comparison post-processing

For comparison questions, `hotpot_dataset.py`:

1. Retrieves evidence texts for each entity from the span list and page text.
2. Adds KG facts for both entities.
3. Uses a KG-based yes/no vote when possible.
4. Uses negation detection to prefer `no` when the combined evidence is strongly negative.
5. Uses year extraction for first/latest-style comparative questions.
6. Adds KG shortest-path evidence between the two entities when available.

Bridge questions use KG shortest-path evidence between query-related entities.

## Models and Transformer Components

The project uses the following named models and pipelines.

| Model                                                           | Used in                                                                                                      | Short description                                                                                                              |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `app.py`                                                                                                   | Multilingual sentence embedding model for the UI path. It is chosen to better align with English, Hindi, and Hinglish queries. |
| `sentence-transformers/all-mpnet-base-v2`                     | `run_custom_qa.py`, `parser/drg_graph.py` default, `parser/enhanced_reasoner.py` default               | General-purpose English sentence embedding model used for sentence and query similarity in the batch pipeline.                 |
| `BAAI/bge-large-en-v1.5`                                      | `hotpot_dataset.py` default CLI model                                                                      | Large embedding model used for HotpotQA evaluation when not overridden.                                                        |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1`            | `parser/span_graph.py` default                                                                             | Span embedding default inside the class. It is typically overridden by the entrypoint model.                                   |
| `dslim/bert-base-NER`                                         | `parser/span_extractor.py`, optional `parser/drg_graph.py` NER mode                                      | Named-entity recognition model used to extract entity spans. The pipeline uses `aggregation_strategy="simple"`.              |
| `deepset/deberta-v3-large-squad2`                             | `parser/answer_selector.py` via `parser/model_cache.py`                                                  | Extractive QA model used to pinpoint the final answer span from the best retrieved passage.                                    |
| `cross-encoder/ms-marco-MiniLM-L-6-v2`                        | `parser/enhanced_reasoner.py` optional reranker                                                            | Pairwise reranker for candidate spans. It is enabled by default in batch paths and disabled in the Streamlit app.              |
| `facebook/bart-large-mnli`                                    | `parser/question_processor.py`                                                                             | Zero-shot classifier used for question type classification.                                                                    |
| `google/flan-t5-base`                                         | `parser/question_processor.py`                                                                             | Text-to-text generator used to decompose complex questions into simpler sub-questions.                                         |
| `en_core_web_trf`                                             | `parser/knowledge_graph.py`, `parser/span_extractor.py`                                                  | Preferred spaCy pipeline for dependency parsing and triple extraction.                                                         |
| `en_core_web_lg`                                              | `parser/knowledge_graph.py` fallback                                                                       | Larger fallback spaCy model when the transformer model is unavailable.                                                         |
| `en_core_web_sm`                                              | `parser/sentence_splitter.py` and fallback in `parser/knowledge_graph.py` / `parser/span_extractor.py` | Small fallback spaCy model for sentence splitting and dependency parsing.                                                      |

### Model loading and caching behavior

`parser/model_cache.py` caches all heavy models in memory:

| Cache                      | Purpose                                                                                                        |
| -------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Sentence transformer cache | Avoids repeated embedding model loads. Embedding models are loaded on CPU in the current cache implementation. |
| QA pipeline cache          | Avoids reloading the extractive QA model. Uses GPU when CUDA is available, otherwise CPU.                      |
| NER pipeline cache         | Avoids reloading the NER model. Uses GPU when CUDA is available, otherwise CPU.                                |

## Configuration, Dependencies, and Runtime Assumptions

### Dependency summary

The main dependency groups are:

| Group                          | Packages                                                                                       |
| ------------------------------ | ---------------------------------------------------------------------------------------------- |
| Numerical and graph processing | `numpy`, `networkx`, `scikit-learn`, `tqdm`, `matplotlib`                            |
| PDF processing                 | `pymupdf`, optional `pdfplumber`, optional `Pillow`, optional `pytesseract`            |
| NLP and models                 | `sentence-transformers`, `transformers`, `torch`, `nltk`, `spacy`                    |
| UI and console                 | `streamlit`, `colorama`                                                                    |
| Optional extras                | `coreferee` for certain spaCy/Python combinations, `torchvision` as listed in requirements |

Version constraints from `requirements.txt`:

| Package                   | Constraint                                 |
| ------------------------- | ------------------------------------------ |
| `numpy`                 | `>=1.20.0`                               |
| `tqdm`                  | `>=4.60.0`                               |
| `networkx`              | `>=2.6`                                  |
| `matplotlib`            | `>=3.5.0`                                |
| `scikit-learn`          | `>=1.0.0`                                |
| `pymupdf`               | `>=1.20.0`                               |
| `sentence-transformers` | `>=2.2.0`                                |
| `transformers`          | `>=4.41.0,<5.0.0`                        |
| `torch`                 | `>=2.0.0`                                |
| `torchvision`           | `>=0.26.0`                               |
| `nltk`                  | `>=3.8.0`                                |
| `spacy`                 | `>=3.7.0,<4.0.0`                         |
| `streamlit`             | `>=1.20.0`                               |
| `colorama`              | `>=0.4.6`                                |
| `Pillow`                | `>=10.0.0`                               |
| `pytesseract`           | `>=0.3.10`                               |
| `pdfplumber`            | `>=0.10.0`                               |
| `coreferee`             | `>=1.4.1` for Python versions below 3.14 |

### Runtime assumptions

1. The pipeline is inference-only; there is no training or fine-tuning code in the repository.
2. Sentence embeddings are loaded through the cache on CPU.
3. QA and NER pipelines can use GPU when available.
4. OCR and table extraction are optional and gracefully disabled when their dependencies are missing.
5. The Streamlit app stores graphs and the reasoner in session state so the PDF does not need to be reprocessed for every question.
6. The batch scripts rebuild graphs per input document or dataset question and do not persist graph state between runs.

## Output Artifacts

### Streamlit app outputs

The Streamlit path exports or shows:

| Artifact                           | Notes                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------ |
| `graphs/span_graph.json`         | Span graph structure.                                                                |
| `graphs/span_graph.png`          | Span graph image.                                                                    |
| `graphs/knowledge_graph.json`    | Knowledge graph export.                                                              |
| `graphs/knowledge_graph.graphml` | Knowledge graph GraphML export.                                                      |
| `graphs/drg_graph.png`           | DRG image.                                                                           |
| On-screen answer cards             | Final answer, confidence, evidence spans, reasoning chains, and retrieval breakdown. |

### Batch QA outputs

`run_custom_qa.py` writes both JSON and TXT reports. The JSON record includes question text, answer, evidence count, evidence spans, retrieval counts, reasoning depth, timing, and evaluation against top evidence.

`hotpot_dataset.py` writes an output JSON and text report that include per-question gold answers, predictions, EM/F1, confidence, KG stats, reasoning depth, and retrieval breakdown.

### Documentation outputs

The repository contains pre-generated visual documentation in `docs/`, including the Mermaid and PlantUML source diagrams and rendered PDF/SVG assets.

## Important Implementation Notes

1. The Streamlit app and the batch scripts do not build identical graph variants. The UI path uses structural DRG edges, while the batch and Hotpot paths rely more on entity overlap in the span graph.
2. The full `SpanGraph.build_graph(...)` method includes entity-node construction, but the current entrypoints do not call it.
3. `extract_document_with_tables` returns tables, but the current QA flows do not consume the table data in downstream graph construction.
4. `PDF_OUTPUT_DIR` is defined in configuration, but the current scripts do not route output files through it.
5. `show_comparisons.py` exposes `--show-context`, but the context printing code is commented out.
6. `QuestionProcessor` fallback classification does not explicitly emit `bridge`; bridge handling depends on the transformer path or on Hotpot-specific heuristics.
7. The answer-selection module docstring mentions a different QA model, but the actual runtime QA pipeline comes from `parser/model_cache.py` and loads `deepset/deberta-v3-large-squad2` by default.
8. Pairwise semantic edge construction in both graph layers is quadratic in node count, so large documents will incur higher preprocessing cost.
9. The code does not contain benchmark tables or trained checkpoints; it only contains runtime evaluation and report generation.

## Appendix: File Inventory

### Top-level files

| File                                                         | Purpose                          |
| ------------------------------------------------------------ | -------------------------------- |
| [app.py](../app.py)                                             | Interactive PDF QA app.          |
| [run_custom_qa.py](../run_custom_qa.py)                         | Single-PDF batch QA runner.      |
| [hotpot_dataset.py](../hotpot_dataset.py)                       | HotpotQA benchmark runner.       |
| [show_comparisons.py](../show_comparisons.py)                   | Comparison question sampler.     |
| [count_question_difficulty.py](../count_question_difficulty.py) | Difficulty counter for HotpotQA. |
| [setup_nltk.py](../setup_nltk.py)                               | NLTK bootstrap script.           |
| [requirements.txt](../requirements.txt)                         | Dependency list.                 |

### Parser package

| File                                                         | Purpose                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------- |
| [parser/__init__.py](../parser/__init__.py)               | Empty initializer.                                            |
| [parser/config.py](../parser/config.py)                         | Config flags for OCR/table extraction/output dir.             |
| [parser/pdf_parser.py](../parser/pdf_parser.py)                 | PDF extraction, OCR fallback, tables, and document structure. |
| [parser/sentence_splitter.py](../parser/sentence_splitter.py)   | Sentence segmentation.                                        |
| [parser/section_utils.py](../parser/section_utils.py)           | Section detection heuristics.                                 |
| [parser/drg_nodes.py](../parser/drg_nodes.py)                   | Sentence node builder.                                        |
| [parser/drg_graph.py](../parser/drg_graph.py)                   | Sentence graph.                                               |
| [parser/span_extractor.py](../parser/span_extractor.py)         | Span extraction.                                              |
| [parser/span_graph.py](../parser/span_graph.py)                 | Span graph.                                                   |
| [parser/knowledge_graph.py](../parser/knowledge_graph.py)       | Knowledge graph.                                              |
| [parser/question_processor.py](../parser/question_processor.py) | Question classification and decomposition.                    |
| [parser/enhanced_reasoner.py](../parser/enhanced_reasoner.py)   | Main reasoner.                                                |
| [parser/advanced_retrieval.py](../parser/advanced_retrieval.py) | Retrieval helpers and edge-weight utilities.                  |
| [parser/answer_selector.py](../parser/answer_selector.py)       | Final answer selection.                                       |
| [parser/evaluator.py](../parser/evaluator.py)                   | Metric functions.                                             |
| [parser/model_cache.py](../parser/model_cache.py)               | Model cache.                                                  |
| [parser/comparison_utils.py](../parser/comparison_utils.py)     | Comparison heuristics.                                        |

### Docs sources and assets

| File                                                                 | Purpose                               |
| -------------------------------------------------------------------- | ------------------------------------- |
| [docs/codebase_flowchart.mmd](codebase_flowchart.mmd)                   | Mermaid flowchart source.             |
| [docs/codebase_flowchart_concise.puml](codebase_flowchart_concise.puml) | PlantUML flowchart source.            |
| [docs/flow_diagram.svg](flow_diagram.svg)                               | Rendered flow diagram asset.          |
| [docs/timeline_diagram.pdf](timeline_diagram.pdf)                       | Rendered timeline diagram asset.      |
| [docs/architecture_diagram.pdf](architecture_diagram.pdf)               | Rendered architecture diagram asset.  |
| [docs/CTRL+ALT+DLT-Mid.pdf](CTRL+ALT+DLT-Mid.pdf)                       | Additional PDF asset present in docs. |

## Verification Status

The workspace checker reported no errors in the inspected source files at the time this report was created.

## Closing Note

Everything above is derived from the source files in this repository. Where the code provides a capability but the main scripts do not use it, the report calls that out explicitly so the implementation view stays accurate.
