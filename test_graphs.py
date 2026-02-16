"""
Comprehensive Test for Enhanced Graph Systems
Tests DRG, Span Graph, and Knowledge Graph with visualization export
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from parser.drg_nodes import build_nodes
from parser.drg_graph import DocumentReasoningGraph
from parser.span_extractor import SpanExtractor
from parser.span_graph import SpanGraph
from parser.kg_builder import KnowledgeGraphBuilder

# Sample test document
test_pages = [
    {
        "page": 1,
        "text": """ASSIGNMENT DETAILS
This assignment is worth 25 marks.
The submission deadline is March 15, 2025 at 11:59 PM.
Students must submit the report by the deadline.
Late submissions will not be accepted except in medical emergencies."""
    },
    {
        "page": 2,
        "text": """GRADING CRITERIA
The report will be graded on three criteria.
Quality of analysis accounts for 10 marks.
Innovation and creativity accounts for 10 marks.
Presentation and formatting accounts for 5 marks.
If the submission is incomplete, a penalty of 50% will be applied."""
    }
]

print("=" * 80)
print("🚀 Testing Enhanced Multi-Level Graph System")
print("=" * 80)

# Step 1: Build sentence nodes
print("\n📝 Step 1: Building sentence nodes...")
nodes = build_nodes(test_pages)
print(f"   ✓ Created {len(nodes)} sentence nodes")

# Step 2: Build DRG
print("\n🔨 Step 2: Building Enhanced DRG...")
drg = DocumentReasoningGraph()
drg.add_nodes(nodes)
drg.compute_embeddings()
drg.add_structural_edges()
drg.add_semantic_edges()
drg.compute_graph_metrics()

# Step 3: Extract spans
print("\n📋 Step 3: Extracting fine-grained spans...")
span_extractor = SpanExtractor()
spans = span_extractor.extract_spans_from_nodes(nodes)
print(f"   ✓ Extracted {len(spans)} spans")

# Step 4: Build Span Graph
print("\n🔗 Step 4: Building Enhanced Span Graph...")
span_graph_builder = SpanGraph()
span_graph_builder.build_graph(spans)

# Step 5: Build Knowledge Graph
print("\n🧠 Step 5: Building Enhanced Knowledge Graph...")
kg_builder = KnowledgeGraphBuilder()
kg = kg_builder.build_kg(spans)

# Step 6: Export for visualization
print("\n" + "=" * 80)
print("💾 Exporting Graphs for Visualization")
print("=" * 80)

# Create output directory
os.makedirs('graphs', exist_ok=True)

# Export Span Graph
print("\n📊 Exporting Span Graph...")
span_graph_builder.export_graph_json('graphs/span_graph.json')
span_graph_builder.export_graph_graphml('graphs/span_graph.graphml')

# Export Knowledge Graph
print("\n🧠 Exporting Knowledge Graph...")
kg_builder.export_graph_json('graphs/kg_graph.json')
kg_builder.export_graph_graphml('graphs/kg_graph.graphml')

# Summary
print("\n" + "=" * 80)
print("📈 Summary Statistics")
print("=" * 80)

print(f"\n🔹 DRG (Sentence-level):")
print(f"   Nodes: {drg.graph.number_of_nodes()}")
print(f"   Edges: {drg.graph.number_of_edges()}")
print(f"   Entities: {len(drg.entity_map)}")

print(f"\n🔹 Span Graph:")
print(f"   Nodes: {span_graph_builder.graph.number_of_nodes()}")
print(f"   Edges: {span_graph_builder.graph.number_of_edges()}")

print(f"\n🔹 Knowledge Graph:")
print(f"   Entities: {len(kg['entities'])}")
print(f"   Relations: {len(kg['relations'])}")
print(f"   Graph nodes: {kg['graph'].number_of_nodes()}")
print(f"   Graph edges: {kg['graph'].number_of_edges()}")

# Entity breakdown
from collections import Counter
entity_types = Counter([e['entity_type'] for e in kg['entities']])
print(f"\n   Entity Types:")
for etype, count in entity_types.most_common():
    print(f"      • {etype}: {count}")

# Relation breakdown
relation_types = Counter([r['type'] for r in kg['relations']])
print(f"\n   Relation Types:")
for rtype, count in relation_types.most_common():
    print(f"      • {rtype}: {count}")

print("\n" + "=" * 80)
print("✅ All Tests Complete!")
print("=" * 80)
print("\n📂 Exported Files:")
print("   • graphs/span_graph.json")
print("   • graphs/span_graph.graphml")
print("   • graphs/kg_graph.json")
print("   • graphs/kg_graph.graphml")
print("\n💡 Use these files with graph visualization tools like:")
print("   - D3.js for web visualization")
print("   - Gephi for network analysis")
print("   - Cytoscape for biological networks")
print("=" * 80)
