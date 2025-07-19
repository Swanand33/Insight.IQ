"""
Basic tests for InsightGenie - ensures core functionality works
"""
import pytest
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all core modules can be imported."""
    try:
        from data_processor import DataProcessor
        from utils import detect_column_purpose
        print("✅ All imports successful")
        assert True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        assert False, f"Import failed: {e}"

def test_data_processor_creation():
    """Test DataProcessor can be created."""
    try:
        from data_processor import DataProcessor
        processor = DataProcessor()
        assert processor is not None
        print("✅ DataProcessor created successfully")
    except Exception as e:
        assert False, f"DataProcessor creation failed: {e}"

def test_ai_analyzer_creation():
    """Test AIAnalyzer can be created with mock key."""
    try:
        from ai_analyzer import AIAnalyzer
        analyzer = AIAnalyzer("test-key")
        assert analyzer is not None
        assert analyzer.model == "gpt-4"
        print("✅ AIAnalyzer created successfully")
    except Exception as e:
        assert False, f"AIAnalyzer creation failed: {e}"

def test_visualizer_creation():
    """Test AutoVisualizer can be created."""
    try:
        from visualization import AutoVisualizer
        visualizer = AutoVisualizer()
        assert visualizer is not None
        assert len(visualizer.color_palette) > 0
        print("✅ AutoVisualizer created successfully")
    except Exception as e:
        assert False, f"AutoVisualizer creation failed: {e}"

if __name__ == "__main__":
    # Run tests directly
    test_imports()
    test_data_processor_creation()
    test_ai_analyzer_creation()
    test_visualizer_creation()
    print("🎉 All basic tests passed!")