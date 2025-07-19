#!/usr/bin/env python3
"""
InsightGenie Project Diagnostic Script
Run this to check what's wrong with your project setup
"""

import os
import sys
from pathlib import Path

def check_file_structure():
    """Check if all required files exist."""
    print("📁 CHECKING FILE STRUCTURE")
    print("=" * 50)
    
    required_files = [
        "src/main.py",
        "src/data_processor.py", 
        "src/ai_analyzer.py",
        "src/visualization.py",
        "src/utils.py",
        "requirements.txt",
        "setup.py"
    ]
    
    required_folders = [
        "src",
        "tests", 
        "prompts",
        "examples",
        "docs"
    ]
    
    # Check folders
    for folder in required_folders:
        if os.path.exists(folder):
            print(f"✅ {folder}/ exists")
        else:
            print(f"❌ {folder}/ MISSING")
    
    # Check files
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} exists ({size} bytes)")
        else:
            print(f"❌ {file} MISSING")

def check_file_contents():
    """Check the content of key files for common issues."""
    print("\n📄 CHECKING FILE CONTENTS")
    print("=" * 50)
    
    files_to_check = {
        "src/utils.py": [99],  # Check line 99 for bracket issue
        "src/visualization.py": [1, 7, 43, 82, 322, 323],  # Check import and syntax lines
        "src/main.py": [1, 185],  # Check imports
        "tests/test_core_functionality.py": [180, 667, 918]  # Check syntax errors
    }
    
    for file_path, line_numbers in files_to_check.items():
        print(f"\n🔍 Checking {file_path}:")
        
        if not os.path.exists(file_path):
            print(f"   ❌ File doesn't exist!")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"   📊 Total lines: {len(lines)}")
            
            for line_num in line_numbers:
                if line_num <= len(lines):
                    line_content = lines[line_num - 1].strip()
                    print(f"   Line {line_num}: {line_content[:80]}...")
                else:
                    print(f"   Line {line_num}: LINE DOESN'T EXIST (file only has {len(lines)} lines)")
                    
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")

def check_imports():
    """Test if Python can import the modules."""
    print("\n🐍 CHECKING PYTHON IMPORTS")
    print("=" * 50)
    
    # Add src to path
    src_path = Path("src").absolute()
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    modules_to_test = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("gradio", "gradio"),
        ("openai", "openai"),
        ("plotly.graph_objects", "plotly"),
        ("data_processor", "data_processor.py"),
        ("ai_analyzer", "ai_analyzer.py"),
        ("visualization", "visualization.py"),
        ("utils", "utils.py")
    ]
    
    for module_name, file_info in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name} - OK")
        except ImportError as e:
            print(f"❌ {module_name} - FAILED: {e}")
        except SyntaxError as e:
            print(f"❌ {module_name} - SYNTAX ERROR: {e}")
        except Exception as e:
            print(f"❌ {module_name} - OTHER ERROR: {e}")

def check_syntax_issues():
    """Check for common syntax issues in Python files."""
    print("\n🔧 CHECKING SYNTAX ISSUES")
    print("=" * 50)
    
    python_files = []
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    for file_path in python_files:
        print(f"\n🔍 Syntax check: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to compile the file
            compile(content, file_path, 'exec')
            print(f"   ✅ Syntax OK")
            
        except SyntaxError as e:
            print(f"   ❌ SYNTAX ERROR at line {e.lineno}: {e.msg}")
            print(f"      Text: {e.text.strip() if e.text else 'N/A'}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")

def show_current_directory():
    """Show what's in the current directory."""
    print("\n📂 CURRENT DIRECTORY CONTENTS")
    print("=" * 50)
    
    current_dir = os.getcwd()
    print(f"Working directory: {current_dir}")
    
    try:
        items = sorted(os.listdir("."))
        for item in items:
            if os.path.isdir(item):
                print(f"📁 {item}/")
            else:
                size = os.path.getsize(item)
                print(f"📄 {item} ({size} bytes)")
    except Exception as e:
        print(f"❌ Error listing directory: {e}")

def check_environment():
    """Check Python environment."""
    print("\n🌍 ENVIRONMENT CHECK")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"PYTHONPATH: {sys.path[:3]}...")  # Show first 3 paths
    
    # Check if in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not in virtual environment")

def generate_fix_suggestions():
    """Generate suggestions to fix common issues."""
    print("\n💡 FIX SUGGESTIONS")
    print("=" * 50)
    
    suggestions = [
        "1. If files are missing, copy the artifact content from my previous messages",
        "2. If imports fail, run: pip install -r requirements.txt",
        "3. If syntax errors exist, check the exact lines mentioned above",
        "4. Make sure you're in the correct directory (should contain src/ folder)",
        "5. If still having issues, share the specific error output with me"
    ]
    
    for suggestion in suggestions:
        print(suggestion)

def main():
    """Run all diagnostic checks."""
    print("🚀 INSIGHTGENIE PROJECT DIAGNOSTIC")
    print("=" * 60)
    print("This script will check what's wrong with your project setup")
    print("=" * 60)
    
    show_current_directory()
    check_environment() 
    check_file_structure()
    check_file_contents()
    check_syntax_issues()
    check_imports()
    generate_fix_suggestions()
    
    print("\n🏁 DIAGNOSTIC COMPLETE")
    print("=" * 50)
    print("Share the output above with me to get specific fixes!")

if __name__ == "__main__":
    main()