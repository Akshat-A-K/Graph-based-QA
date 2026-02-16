"""
Quick test to showcase Enhanced DRG improvements
"""

from parser.drg_nodes import build_nodes
from parser.drg_graph import DocumentReasoningGraph

# Sample test data
test_pages = [
    {
        "page": 1,
        "text": """ASSIGNMENT DETAILS
This assignment is worth 25 marks.
The submission deadline is March 15, 2025.
Students must submit the report by the deadline."""
    },
    {
        "page": 2,
        "text": """GRADING CRITERIA
The report will be graded on three criteria.
Quality accounts for 10 marks.
Innovation accounts for 10 marks.
Presentation accounts for 5 marks."""
    }
]

print("=" * 70)
print("🚀 Testing Enhanced DRG System")
print("=" * 70)

# Build nodes
print("\n📝 Building sentence nodes...")
nodes = build_nodes(test_pages)
print(f"   ✓ Created {len(nodes)} sentence nodes")

# Build enhanced DRG
print("\n🔨 Building enhanced Document Reasoning Graph...")
drg = DocumentReasoningGraph()
graph = drg.build_graph(nodes)

# Show node details
print("\n" + "=" * 70)
print("📊 Sample Node Analysis")
print("=" * 70)

sample_node = list(graph.nodes())[0]
node_data = graph.nodes[sample_node]

print(f"\nNode ID: {sample_node}")
print(f"   Text: {node_data['text'][:60]}...")
print(f"   Page: {node_data['page']}")
print(f"   Section: {node_data['section']}")
print(f"   Importance: {node_data.get('importance', 0):.3f}")
print(f"   Length (words): {node_data.get('length', 0)}")
print(f"   Entities: {node_data.get('entities', [])}")
print(f"   Degree: {node_data.get('degree', 0)}")
if 'pagerank' in node_data:
    print(f"   PageRank: {node_data['pagerank']:.4f}")

# Show edge analysis
print("\n" + "=" * 70)
print("🔗 Edge Analysis")
print("=" * 70)

from collections import Counter
edge_types = Counter()
for u, v, data in graph.edges(data=True):
    edge_types[data.get('type', 'unknown')] += 1

print("\nEdge Type Distribution:")
for edge_type, count in edge_types.most_common():
    print(f"   • {edge_type:15} : {count:4} edges")

# Show entity connections
print("\n" + "=" * 70)
print("🏷️  Entity Coreference Map")
print("=" * 70)

if drg.entity_map:
    for entity, node_ids in list(drg.entity_map.items())[:5]:
        print(f"\n   {entity}")
        print(f"      Connected nodes: {node_ids}")
else:
    print("   No entities detected in this sample")

print("\n" + "=" * 70)
print("✅ Enhanced DRG Test Complete!")
print("=" * 70)
