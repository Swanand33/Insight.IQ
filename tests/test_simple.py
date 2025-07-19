"""
Simple working tests for InsightGenie
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test basic imports work."""
    print("Testing imports...")
    
    try:
        from data_processor import DataProcessor
        print("✅ DataProcessor imported")
    except Exception as e:
        print(f"❌ DataProcessor failed: {e}")
        return False
    
    try:
        from ai_analyzer import AIAnalyzer
        print("✅ AIAnalyzer imported") 
    except Exception as e:
        print(f"❌ AIAnalyzer failed: {e}")
        return False
        
    try:
        from utils import detect_column_purpose
        print("✅ Utils imported")
    except Exception as e:
        print(f"❌ Utils failed: {e}")
        return False
    
    print("🎉 All imports successful!")
    return True

if __name__ == "__main__":
    test_imports()