#!/usr/bin/env python3
"""
Quick setup script for InsightGenie
Run this to set up your environment and test the installation
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command):
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {command}")
            return True
        else:
            print(f"❌ {command}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {command}")
        print(f"Error: {str(e)}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False

def install_requirements():
    """Install required packages."""
    print("\n📦 Installing requirements...")
    return run_command(f"{sys.executable} -m pip install -r requirements.txt")

def check_api_key():
    """Check if OpenAI API key is set."""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and api_key.startswith('sk-'):
        print("✅ OpenAI API key found")
        return True
    else:
        print("⚠️  OpenAI API key not found")
        print("   Set your API key with: export OPENAI_API_KEY='your-key-here'")
        print("   Or create a .env file with: OPENAI_API_KEY=your-key-here")
        return False

def create_test_data():
    """Create sample test data for demonstration."""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    print("\n📊 Creating sample test data...")
    
    # Sample sales data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    sales_data = {
        'date': dates,
        'revenue': np.random.normal(50000, 15000, len(dates)).round(2),
        'units_sold': np.random.poisson(200, len(dates)),
        'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], len(dates)),
        'customer_satisfaction': np.random.uniform(3.5, 5.0, len(dates)).round(1),
        'marketing_spend': np.random.normal(5000, 1500, len(dates)).round(2)
    }
    
    df = pd.DataFrame(sales_data)
    
    # Add some trends and seasonality
    df['revenue'] *= (1 + 0.5 * np.sin(2 * np.pi * df.index / 365))  # Seasonal pattern
    df['revenue'] += df.index * 10  # Growth trend
    df['revenue'] = df['revenue'].round(2)
    
    # Add some missing values and outliers for realism
    missing_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[missing_indices, 'customer_satisfaction'] = np.nan
    
    # Add outliers
    outlier_indices = np.random.choice(df.index, size=5, replace=False)
    df.loc[outlier_indices, 'revenue'] *= 3
    
    # Save sample data
    Path('examples').mkdir(exist_ok=True)
    df.to_excel('examples/sample_sales_data.xlsx', index=False)
    print("✅ Created examples/sample_sales_data.xlsx")
    
    # Create HR sample data
    hr_data = {
        'employee_id': range(1, 501),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], 500),
        'salary': np.random.normal(75000, 20000, 500).round(0),
        'years_experience': np.random.exponential(5, 500).round(1),
        'performance_score': np.random.uniform(2.5, 5.0, 500).round(1),
        'training_hours': np.random.poisson(40, 500),
        'remote_work_days': np.random.choice([0, 1, 2, 3, 4, 5], 500),
        'job_satisfaction': np.random.uniform(3.0, 5.0, 500).round(1)
    }
    
    hr_df = pd.DataFrame(hr_data)
    hr_df.to_csv('examples/sample_hr_data.csv', index=False)
    print("✅ Created examples/sample_hr_data.csv")
    
    return True

def test_import():
    """Test if all modules can be imported."""
    print("\n🧪 Testing imports...")
    
    try:
        from data_processor import DataProcessor
        print("✅ data_processor imported")
    except Exception as e:
        print(f"❌ data_processor import failed: {e}")
        return False
    
    try:
        from ai_analyzer import AIAnalyzer
        print("✅ ai_analyzer imported")
    except Exception as e:
        print(f"❌ ai_analyzer import failed: {e}")
        return False
    
    try:
        import gradio as gr
        print("✅ gradio imported")
    except Exception as e:
        print(f"❌ gradio import failed: {e}")
        return False
    
    return True

def main():
    """Main setup function."""
    print("🚀 InsightGenie Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Python version too old")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed: Could not install requirements")
        sys.exit(1)
    
    # Test imports
    if not test_import():
        print("\n❌ Setup failed: Import errors")
        sys.exit(1)
    
    # Check API key
    api_key_ok = check_api_key()
    
    # Create test data
    create_test_data()
    
    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    print("\n📋 Next Steps:")
    print("1. Run the app: python main.py")
    print("2. Open browser to: http://localhost:7860")
    print("3. Upload examples/sample_sales_data.xlsx to test")
    
    if not api_key_ok:
        print("\n⚠️  Note: Set your OpenAI API key to enable AI analysis")
        print("   export OPENAI_API_KEY='your-key-here'")
    
    print("\n🎯 Test with sample data:")
    print("   - examples/sample_sales_data.xlsx (financial analysis)")
    print("   - examples/sample_hr_data.csv (operational analysis)")

if __name__ == "__main__":
    main()