"""
Setup script to download required NLTK data
Run this once after installing requirements
"""

import nltk
import sys

def download_nltk_data():
    """Download required NLTK datasets"""
    print("Downloading NLTK data...")
    
    try:
        # Download stopwords
        print("  - Downloading stopwords...")
        nltk.download('stopwords', quiet=False)
        
        # Download punkt (for better sentence tokenization if needed)
        print("  - Downloading punkt...")
        nltk.download('punkt', quiet=False)
        
        print("\nNLTK data downloaded successfully!")
        print("   You can now run the QA system.")
        
        # Test import
        from nltk.corpus import stopwords
        stop_words = stopwords.words('english')
        print(f"   Loaded {len(stop_words)} English stopwords")
        
        return True
        
    except Exception as e:
        print(f"\nError downloading NLTK data: {e}")
        print("   Please check your internet connection and try again.")
        return False


if __name__ == "__main__":
    success = download_nltk_data()
    sys.exit(0 if success else 1)
