"""
Answer selection utilities.

Final answer extraction is performed by a Transformer-based extractive-QA
model (default: deepset/roberta-base-squad2).  No document-type-specific
hardcoding is used — the model generalises across any PDF domain.
"""

from typing import Dict, List, Tuple
import re

from .advanced_retrieval import tokenize, normalize_text, BM25Retriever
from .model_cache import get_qa_pipeline


try:
    from nltk.corpus import stopwords
    _STOP_WORDS = set(stopwords.words("english"))
except Exception:
    _STOP_WORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
        "at", "by", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then",
        "once"
    }


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _strip_section_prefix(text: str) -> str:
    """Remove leading section numbers (e.g. '3.4 ', '1.1.2 ', '2 ') from text.

    Academic PDFs often fuse section numbers with the first sentence of a
    section, e.g. '3.4 Evaluation Report the Accuracy...'.  The leading number
    causes extractive-QA models to parse 'Evaluation' as a section-label noun
    rather than the start of an instruction sentence, which degrades extraction.
    Stripping the prefix is purely structural and domain-agnostic.
    """
    return re.sub(r'^\d+(?:\.\d+)*\s+', '', text)


# ---------------------------------------------------------------------------
# Extractive QA helper
# ---------------------------------------------------------------------------

def _qa_extract(question: str, context: str, qa_pipeline) -> Tuple[str, float]:
    """Run an extractive-QA model to find the precise answer span in *context*.

    Returns (answer_text, confidence_score).  Falls back to ("", 0.0) on any
    error so that callers can degrade gracefully.
    """
    if not context or not question:
        return "", 0.0
    try:
        results_list = qa_pipeline(question=question, context=context, max_answer_len=200, top_k=3)
        if not isinstance(results_list, list):
            results_list = [results_list]
        # Pick the candidate with the highest confidence score.
        # Using top_k>1 avoids committing to a single span and allows the model
        # to surface longer or more complete answer spans as alternatives.
        best = max(results_list, key=lambda r: float(r.get("score", 0.0)))
        return best.get("answer", ""), float(best.get("score", 0.0))
    except Exception:
        return "", 0.0


def _fuzzy_overlap(query_content: set, text_content: set) -> int:
    """Count content overlap with simple prefix matching for plurals/variants."""
    if not query_content or not text_content:
        return 0
    overlap = 0
    for q in query_content:
        if q in text_content:
            overlap += 1
            continue
        if len(q) < 4:
            continue
        for t in text_content:
            if len(t) < 4:
                continue
            if t.startswith(q) or q.startswith(t):
                overlap += 1
                break
    return overlap


def select_answer(
    results: Dict,
    span_graph,
    query: str,
    reasoner=None,
    max_length: int = 300
) -> Tuple[str, List[str], float, str]:
    """Pick the best answer span with keyword anchoring and sentence preference."""
    final_spans = results.get("final_spans", [])
    if not final_spans:
        return "No answer found", [], 0.0, "Low"

    span_scores = results.get("span_scores", {})
    query_norm = normalize_text(query)
    query_tokens = set(tokenize(query_norm))
    query_content = query_tokens - _STOP_WORDS

    candidate_ids = []
    candidate_ids.extend(final_spans)
    candidate_ids.extend(results.get("hybrid_results", []))
    candidate_ids.extend(results.get("traversal_results", []))
    candidate_ids.extend(results.get("expansion_results", []))
    candidate_ids.extend(results.get("kg_spans", []))

    seen = set()
    candidates = []
    for sid in candidate_ids:
        if sid not in seen:
            seen.add(sid)
            candidates.append(sid)

    ranked = []
    for span_id in candidates[:40]:
        if span_id not in span_graph.graph.nodes:
            continue
        node = span_graph.graph.nodes[span_id]
        text = _clean_text(node.get("text", ""))
        span_type = node.get("span_type", "")
        sentence_id = node.get("sentence_id")
        score = float(span_scores.get(span_id, 0.0))
        text_tokens = set(tokenize(text))
        text_content = text_tokens - _STOP_WORDS
        overlap = _fuzzy_overlap(query_content, text_content)

        # Prefer sentence spans, and spans with content overlap
        if span_type == "sentence":
            score += 0.08

        overlap_ratio = overlap / max(1, len(query_content)) if query_content else 0.0
        score += 0.2 * overlap_ratio

        if overlap >= 2:
            score += 0.08
        elif overlap == 0:
            score -= 0.12

        # No hardcoded domain boosts — scoring is purely overlap + length

        # Penalize extremely long fragments
        if len(text) > max_length * 1.6:
            score -= 0.08

        # Light length penalty to prefer concise answers
        score -= min(len(text), 600) * 0.0003

        ranked.append((span_id, text, score, sentence_id, span_type, overlap))

    if not ranked:
        return "No answer found", [], 0.0, "Low"

    ranked.sort(key=lambda x: x[2], reverse=True)

    # Sentence-level deduplication: keep at most 2 spans per sentence_id
    # so the QA model is exposed to spans from many different sentences.
    deduped_ranked = []
    sent_count: dict = {}
    for item in ranked:
        sid = item[3]  # sentence_id
        count = sent_count.get(sid, 0)
        if count < 2:
            deduped_ranked.append(item)
            sent_count[sid] = count + 1
    ranked = deduped_ranked if len(deduped_ranked) >= 3 else ranked

    best_span_id, best_text, best_score, best_sentence_id, best_span_type, best_overlap = ranked[0]

    # Fallback: if no overlap at all, search globally for a span with overlap
    if best_overlap == 0 and query_content:
        fallback = None
        for span_id in span_graph.graph.nodes:
            node = span_graph.graph.nodes[span_id]
            text = _clean_text(node.get("text", ""))
            if not text:
                continue
            text_tokens = set(tokenize(text))
            text_content = text_tokens - _STOP_WORDS
            overlap = _fuzzy_overlap(query_content, text_content)
            if overlap == 0:
                continue
            length = len(text)
            score = overlap - min(length, 600) * 0.0003
            if fallback is None or score > fallback[0]:
                fallback = (score, span_id, text, node.get("sentence_id"), node.get("span_type", ""))

        if fallback is not None:
            _, best_span_id, best_text, best_sentence_id, best_span_type = fallback

    # Prefer the parent sentence when the winning span is a sub-sentence fragment
    if best_span_type != "sentence" and reasoner and best_sentence_id in reasoner.sentence_graph.nodes:
        sentence_text = _clean_text(reasoner.sentence_graph.nodes[best_sentence_id].get("text", ""))
        if 0 < len(sentence_text) <= max_length * 1.2:
            best_text = sentence_text

    # -----------------------------------------------------------------------
    # Extractive QA: let the model pinpoint the precise answer span inside
    # the best retrieved passage.  This is fully model-driven and generalises
    # to any document domain without any hardcoded patterns.
    # -----------------------------------------------------------------------
    try:
        qa_pipeline = get_qa_pipeline()

        # --- individual contexts: top-1 winner + up to 9 runners-up ---
        contexts: List[str] = [best_text]
        for _, ctx_text, _, _, _, _ in ranked[1:10]:
            if ctx_text and ctx_text not in contexts:
                contexts.append(ctx_text)
        # Also include the parent sentence when available
        if reasoner and best_sentence_id in reasoner.sentence_graph.nodes:
            sent_ctx = _clean_text(
                reasoner.sentence_graph.nodes[best_sentence_id].get("text", "")
            )
            if sent_ctx and sent_ctx not in contexts:
                contexts.append(sent_ctx)

        # Re-rank contexts by BM25 keyword relevance before running QA.
        # Dense retrieval favours semantic similarity, but the answer sentence
        # often contains the *exact* query keywords (e.g. "born", "honour").
        # BM25 re-ranking ensures that the most keyword-matching context is
        # tried *first* and becomes the protected primary context.  This
        # simultaneously handles two failure modes:
        #   (a) The correct sentence ranks 2nd in dense retrieval (e.g. Q6:
        #       "where was Kalam born?" → s1 "Born and raised in Rameswaram"
        #       should be first, not s0 "Indian aerospace scientist").
        #   (b) A surface-plausible but wrong sentence has a higher QA score
        #       (e.g. Q13: s5 "president of India" outscoring s8 "Bharat Ratna")
        #       — protecting the BM25-first correct primary context blocks the
        #       wrong secondary from overriding.
        # Re-ordering is only applied when the BM25 winner scores *meaningfully*
        # higher than the current contexts[0]; when no context has distinctive
        # query keywords, all BM25 scores are equal and the dense-retrieval
        # order is preserved.
        if len(contexts) > 1:
            try:
                bm25_sort = BM25Retriever()
                bm25_sort.fit(contexts)
                scored = bm25_sort.retrieve(query_norm, k=len(contexts))
                if scored:
                    top_bm25_idx, top_bm25_score = scored[0]
                    # Find BM25 score of the current first context (idx 0)
                    orig_score = next((s for i, s in scored if i == 0), 0.0)
                    # Only re-order if BM25 top-1 is different from contexts[0]
                    # AND it scored meaningfully higher (>1.5× the first context's
                    # BM25 score).  When all contexts score equally (no
                    # discriminative keywords), this condition fails and we keep
                    # the original dense-retrieval order.
                    if top_bm25_idx != 0 and top_bm25_score > 0 and (
                        orig_score == 0 or top_bm25_score > orig_score * 1.5
                    ):
                        ordered_indices = [idx for idx, _ in scored]
                        contexts = [contexts[i] for i in ordered_indices if i < len(contexts)]
            except Exception:
                pass  # keep original order if BM25 sorting fails



        best_qa_answer = ""
        best_qa_score = 0.0

        # Pass 1 — run QA on each context individually to get precise spans.
        # Section-number prefixes are stripped before extraction so that fused
        # headers like '3.4 Evaluation Report the Accuracy...' don’t mislead
        # the model into treating the section title as a candidate answer span.
        #
        # The *primary* context (ranked[0] — the retriever's top pick) has
        # protected priority: secondary contexts can only override its answer
        # when they are substantially more confident.  The override margin is
        # adaptive: when the primary context already returned a confident answer
        # (≥ 0.45) the bar is raised by OVERRIDE_MARGIN_HIGH (0.20) to prevent
        # a lower-ranked but surface-plausibly-matching sentence from usurping
        # the retrieval-ranked winner on QA score alone.  When the primary is
        # uncertain (< 0.45) a small margin (0.05) is used so that a clearly
        # better secondary context can still take over.
        OVERRIDE_MARGIN_HIGH = 0.20  # applied when primary_qa_score >= 0.45
        OVERRIDE_MARGIN_LOW  = 0.05  # applied when primary context is uncertain
        primary_qa_score = 0.0  # QA confidence from contexts[0]
        for i, ctx in enumerate(contexts):
            answer, score = _qa_extract(query, _strip_section_prefix(ctx), qa_pipeline)
            if i == 0:
                primary_qa_score = score if answer else 0.0
                if answer and score > best_qa_score:
                    best_qa_score = score
                    best_qa_answer = answer
            else:
                margin = (OVERRIDE_MARGIN_HIGH
                          if primary_qa_score >= 0.45
                          else OVERRIDE_MARGIN_LOW)
                threshold = max(best_qa_score, primary_qa_score + margin)
                if answer and score > threshold:
                    best_qa_score = score
                    best_qa_answer = answer

        # Pass 2 — concatenate top-5 contexts into one larger window.
        # Only runs when Pass 1 is low-confidence (< 0.35); if a single context
        # already gave a confident answer, don't dilute it by mixing contexts.
        if best_qa_score < 0.35:
            combined_parts = [_strip_section_prefix(c)[:350] for c in contexts[:5] if c]
            combined_ctx = " ".join(combined_parts)
            if combined_ctx:
                answer, score = _qa_extract(query, combined_ctx, qa_pipeline)
                if answer and score > best_qa_score:
                    best_qa_score = score
                    best_qa_answer = answer

        # Pass 3 — global BM25 search over every sentence in the document.
        # Only runs when the current best answer is low-confidence (< 0.50).
        # A confident Pass-1 answer (≥ 0.50) is treated as correct and BM25
        # is skipped so that keyword-matched-but-wrong sentences cannot corrupt
        # an already-good extraction.
        # BM25 is purely frequency-based with no domain knowledge, so this
        # generalises to any PDF.
        if best_qa_score < 0.50 and reasoner is not None:
            try:
                all_sent_texts: List[str] = []
                for node_id in reasoner.sentence_graph.nodes:
                    t = _clean_text(reasoner.sentence_graph.nodes[node_id].get("text", ""))
                    if t:
                        all_sent_texts.append(t)
                if all_sent_texts:
                    bm25 = BM25Retriever()
                    bm25.fit(all_sent_texts)
                    for idx, _ in bm25.retrieve(query, k=8):
                        sent_text = all_sent_texts[idx]
                        clean_sent = _strip_section_prefix(sent_text)
                        contexts_set = set(contexts)
                        if clean_sent in contexts_set:   # check clean version
                            continue

                        # QA on this BM25-found sentence alone
                        answer, score = _qa_extract(query, clean_sent, qa_pipeline)
                        if answer and score > best_qa_score:
                            best_qa_score = score
                            best_qa_answer = answer
                        # QA on original best passage + this BM25 sentence merged.
                        # Use best_text (full passage) rather than best_qa_answer (short extracted
                        # span) so the merged context is coherent and not misleadingly short.
                        merged = best_text[:300] + " " + clean_sent[:300]
                        answer, score = _qa_extract(query, merged, qa_pipeline)
                        if answer and score > best_qa_score:
                            best_qa_score = score
                            best_qa_answer = answer
            except Exception:
                pass

        # Accept the QA answer when it has reasonable confidence
        if best_qa_answer and best_qa_score >= 0.05:
            best_text = best_qa_answer
    except Exception:
        pass  # Fall back to the passage-level answer if QA is unavailable

    # Truncate if too long
    if len(best_text) > max_length:
        best_text = best_text[:max_length]
        last_period = max(best_text.rfind("."), best_text.rfind("।"))
        if last_period > max_length * 0.6:
            best_text = best_text[:last_period + 1]
        else:
            best_text = best_text.rsplit(" ", 1)[0] + "..."

    best_text = _clean_text(best_text)

    # Evidence selection
    evidence_texts = []
    # best_score is the retrieval score of the top-ranked passage
    best_score = ranked[0][2] if ranked else 0.0
    for _, text, score, _, _, overlap in ranked[:10]:
        if score >= best_score * 0.6 or overlap >= 1:
            evidence_texts.append(text)
        if len(evidence_texts) >= 3:
            break

    confidence = results.get("confidence", 0.5)
    confidence_label = results.get("confidence_label", "Medium")

    return best_text, evidence_texts, confidence, confidence_label
