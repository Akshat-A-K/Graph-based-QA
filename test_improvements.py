"""
Quick validation test for v2.1 improvements
Tests NLTK integration and new filtering logic
"""

def test_nltk_stopwords():
    """Test NLTK stopwords are loaded correctly"""
    print("🧪 Testing NLTK stopwords...")
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        print(f"   ✅ Loaded {len(stop_words)} stopwords")
        assert len(stop_words) > 100, "Too few stopwords"
        assert 'the' in stop_words and 'is' in stop_words
        print("   ✅ Common stopwords present")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_stopword_filtering():
    """Test stopword filtering logic"""
    print("\n🧪 Testing stopword filtering...")
    try:
        from parser.advanced_retrieval import tokenize
        from nltk.corpus import stopwords
        
        stop_words = set(stopwords.words('english'))
        
        # Test case 1: Common words should be filtered
        query_tokens = set(tokenize("what is the deadline"))
        text_tokens = set(tokenize("the output is ready"))
        
        query_content = query_tokens - stop_words
        text_content = text_tokens - stop_words
        overlap = len(query_content & text_content)
        
        print(f"   Query content words: {query_content}")
        print(f"   Text content words: {text_content}")
        print(f"   Meaningful overlap: {overlap}")
        
        assert overlap == 0, "Should have no meaningful overlap"
        print("   ✅ Stopword filtering works correctly")
        
        # Test case 2: Meaningful words should match
        query_tokens2 = set(tokenize("submission deadline"))
        text_tokens2 = set(tokenize("the deadline is 23:59"))
        
        query_content2 = query_tokens2 - stop_words
        text_content2 = text_tokens2 - stop_words
        overlap2 = len(query_content2 & text_content2)
        
        print(f"\n   Query content words: {query_content2}")
        print(f"   Text content words: {text_content2}")
        print(f"   Meaningful overlap: {overlap2}")
        
        assert overlap2 >= 1, "Should have meaningful overlap"
        print("   ✅ Meaningful word matching works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_improved_span_extraction():
    """Test improved span extraction logic"""
    print("\n🧪 Testing improved span extraction...")
    try:
        from parser.span_extractor import SpanExtractor
        
        extractor = SpanExtractor()
        
        # Test sentence
        sentence = "The submission deadline is 15th February 2026 at 23:59."
        
        # Extract spans
        spans = extractor.extract_spans_from_sentence(
            sentence=sentence,
            sentence_id=0,
            page=1,
            section="Deadlines",
            start_span_id=0
        )
        
        print(f"   Extracted {len(spans)} spans:")
        for i, span in enumerate(spans):
            print(f"     {i}. [{span.span_type}] {span.text[:60]}...")
        
        # Should have full sentence + deadline span
        assert len(spans) >= 2, "Should extract at least 2 spans"
        assert spans[0].span_type == "sentence", "First span should be full sentence"
        assert any("deadline" in s.text.lower() for s in spans), "Should extract deadline"
        
        print("   ✅ Span extraction improved")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Test all critical imports work"""
    print("\n🧪 Testing critical imports...")
    try:
        print("   Testing NLTK...")
        import nltk
        from nltk.corpus import stopwords
        
        print("   Testing parser modules...")
        from parser.enhanced_reasoner import EnhancedHybridReasoner
        from parser.advanced_retrieval import tokenize, normalize_text
        from parser.span_extractor import SpanExtractor
        
        print("   Testing app...")
        # Don't import streamlit (requires display)
        
        print("   ✅ All imports successful")
        return True
        
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 Graph-Based QA v2.1 - Validation Tests")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("NLTK Stopwords", test_nltk_stopwords()))
    results.append(("Stopword Filtering", test_stopword_filtering()))
    results.append(("Span Extraction", test_improved_span_extraction()))
    results.append(("Critical Imports", test_imports()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! System is ready.")
        print("  Run: streamlit run app.py")
        return 0
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed. Please check errors above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
