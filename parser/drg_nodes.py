from typing import List, Dict
from .sentence_splitter import split_into_sentences
from .section_utils import detect_section


def build_nodes(pages: List[Dict]) -> List[Dict]:
    """
    Convert pages -> sentence nodes
    """

    nodes = []
    node_id = 0
    current_section = "GLOBAL"

    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]

        lines = text.split("\n")

        for line in lines:
            sec = detect_section(line)
            if sec:
                current_section = sec

        sentences = split_into_sentences(text)

        for idx, sent in enumerate(sentences):
            node = {
                "node_id": node_id,
                "text": sent,
                "page": page_num,
                "section": current_section,
                "sent_index": idx
            }

            nodes.append(node)
            node_id += 1

    return nodes
