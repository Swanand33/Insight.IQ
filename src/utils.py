import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
import re
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

def detect_column_purpose(column_name: str, sample_values: pd.Series) -> str:
    """
    Detect the business purpose of a column based on name and sample values.
    
    Args:
        column_name: Name of the column
        sample_values: Sample values from the column
        
    Returns:
        String indicating the likely purpose of the column
    """
    col_lower = column_name.lower()
    
    # Financial columns
    financial_keywords = ['revenue', 'cost', 'price', 'amount', 'total', 'sales', 'profit', 'margin', 'expense']
    if any(keyword in col_lower for keyword in financial_keywords):
        return 'financial'
    
    # Date/time columns
    date_keywords = ['date', 'time', 'created', 'updated', 'timestamp', 'year', 'month']
    if any(keyword in col_lower for keyword in date_keywords):
        return 'temporal'
    
    # ID columns
    id_keywords = ['id', 'key', 'index', 'reference', 'ref']
    if any(keyword in col_lower for keyword in id_keywords):
        return 'identifier'
    
    # Geographic columns
    geo_keywords = ['country', 'state', 'city', 'region', 'location', 'address', 'zip', 'postal']
    if any(keyword in col_lower for keyword in geo_keywords):
        return 'geographic'
    
    # Categorical columns
    category_keywords = ['category', 'type', 'status', 'grade', 'level', 'group', 'class']
    if any(keyword in col_lower for keyword in category_keywords):
        return 'categorical'
    
    # Performance metrics
    performance_keywords = ['score', 'rating', 'performance', 'efficiency', 'productivity', 'quality']
    if any(keyword in col_lower for keyword in performance_keywords):
        return 'performance'
    
    # Count/quantity columns
    count_keywords = ['count', 'quantity', 'number', 'total', 'sum']
    if any(keyword in col_lower for keyword in count_keywords):
        return 'quantity'
    
    # Analyze sample values for additional context
    if not sample_values.empty:
        # Check if values look like percentages
        if is_percentage_column(sample_values):
            return 'percentage'
        
        # Check if values look like currencies
        if is_currency_column(sample_values):
            return 'financial'
        
        # Check if values are binary (0/1, True/False)
        if is_binary_column(sample_values):
            return 'binary'
    
    return 'general'

def is_percentage_column(series: pd.Series) -> bool:
    """Check if a column contains percentage values."""
    try:
        # Check for percentage symbols
        if series.dtype == 'object':
            sample_str = str(series.dropna().iloc[0]) if len(series.dropna()) > 0 else ""
            if '%' in sample_str:
                return True
        
        # Check for decimal values between 0 and 1
        if pd.api.types.is_numeric_dtype(series):
            numeric_values = series.dropna()
            if len(numeric_values) > 0:
                return (numeric_values >= 0).all() and (numeric_values <= 1).all() and numeric_values.max() < 1
    except:
        pass
    
    return False

def is_currency_column(series: pd.Series) -> bool:
    """Check if a column contains currency values."""
    try:
        if series.dtype == 'object':
            sample_str = str(series.dropna().iloc[0]) if len(series.dropna()) > 0 else ""
            currency_symbols = ['$', '€', '£', '¥', '₹']
            return any(symbol in sample_str for symbol in currency_symbols)
    except:
        pass
    
    return False

def is_binary_column(series: pd.Series) -> bool:
    """Check if a column contains binary values."""
    try:
        unique_values = series.dropna().unique()
        if len(unique_values) == 2:
            # Check for common binary patterns
            binary_patterns = [
                {0, 1}, {'0', '1'}, 
                {True, False}, {'True', 'False'}, {'true', 'false'},
                {'Yes', 'No'}, {'yes', 'no'}, {'Y', 'N'}, {'y', 'n'}
            ]
            
            unique_set = set(unique_values)
            return any(unique_set == pattern for pattern in binary_patterns)
    except:
        pass
    
    return False

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize column names."""
    df = df.copy()
    
    # Clean column names
    df.columns = df.columns.str.strip()  # Remove whitespace
    df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)  # Remove special chars
    df.columns = df.columns.str.replace(r'\s+', '_', regex=True)  # Replace spaces with underscores
    df.columns = df.columns.str.lower()  # Convert to lowercase
    
    # Handle duplicate column names
    seen = {}
    new_columns = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_columns.append(col)
    
    df.columns = new_columns
    return df

def detect_outliers(series: pd.Series, method: str = 'iqr') -> Dict[str, Any]:
    """
    Detect outliers in a numeric series.
    
    Args:
        series: Numeric pandas Series
        method: Method to use ('iqr', 'zscore', 'modified_zscore')
        
    Returns:
        Dictionary with outlier information
    """
    if not pd.api.types.is_numeric_dtype(series):
        return {'outliers': [], 'method': method, 'count': 0}
    
    series_clean = series.dropna()
    
    if len(series_clean) < 4:  # Need minimum data for outlier detection
        return {'outliers': [], 'method': method, 'count': 0}
    
    outliers = []
    
    if method == 'iqr':
        Q1 = series_clean.quantile(0.25)
        Q3 = series_clean.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series_clean[(series_clean < lower_bound) | (series_clean > upper_bound)]
    
    elif method == 'zscore':
        z_scores = np.abs((series_clean - series_clean.mean()) / series_clean.std())
        outliers = series_clean[z_scores > 3]
    
    elif method == 'modified_zscore':
        median = series_clean.median()
        mad = np.median(np.abs(series_clean - median))
        modified_z_scores = 0.6745 * (series_clean - median) / mad
        outliers = series_clean[np.abs(modified_z_scores) > 3.5]
    
    return {
        'outliers': outliers.tolist(),
        'outlier_indices': outliers.index.tolist(),
        'method': method,
        'count': len(outliers),
        'percentage': len(outliers) / len(series_clean) * 100 if len(series_clean) > 0 else 0
    }

def analyze_data_distribution(series: pd.Series) -> Dict[str, Any]:
    """Analyze the distribution characteristics of a series."""
    if not pd.api.types.is_numeric_dtype(series):
        return {'type': 'non_numeric'}
    
    series_clean = series.dropna()
    
    if len(series_clean) == 0:
        return {'type': 'empty'}
    
    # Basic statistics
    stats = {
        'type': 'numeric',
        'count': len(series_clean),
        'mean': series_clean.mean(),
        'median': series_clean.median(),
        'std': series_clean.std(),
        'min': series_clean.min(),
        'max': series_clean.max(),
        'range': series_clean.max() - series_clean.min(),
        'skewness': series_clean.skew(),
        'kurtosis': series_clean.kurtosis()
    }
    
    # Distribution shape
    if abs(stats['skewness']) < 0.5:
        stats['distribution_shape'] = 'symmetric'
    elif stats['skewness'] > 0.5:
        stats['distribution_shape'] = 'right_skewed'
    else:
        stats['distribution_shape'] = 'left_skewed'
    
    # Kurtosis interpretation
    if stats['kurtosis'] > 3:
        stats['tail_behavior'] = 'heavy_tailed'
    elif stats['kurtosis'] < 3:
        stats['tail_behavior'] = 'light_tailed'
    else:
        stats['tail_behavior'] = 'normal_tailed'
    
    return stats

def format_number(value: Union[int, float], format_type: str = 'auto') -> str:
    """Format numbers for display."""
    if pd.isna(value):
        return "N/A"
    
    if format_type == 'currency':
        return f"${value:,.2f}"
    elif format_type == 'percentage':
        return f"{value:.1%}"
    elif format_type == 'integer':
        return f"{int(value):,}"
    elif format_type == 'decimal':
        return f"{value:.2f}"
    else:  # auto
        if abs(value) >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            return f"{value/1_000:.1f}K"
        elif isinstance(value, float) and value != int(value):
            return f"{value:.2f}"
        else:
            return f"{int(value):,}"

def calculate_correlation_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate correlation insights for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return {'message': 'Insufficient numeric columns for correlation analysis'}
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Find strong correlations (exclude diagonal)
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:  # Strong correlation threshold
                strong_correlations.append({
                    'column1': corr_matrix.columns[i],
                    'column2': corr_matrix.columns[j],
                    'correlation': corr_value,
                    'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                })
    
    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'strong_correlations': strong_correlations,
        'summary': f"Found {len(strong_correlations)} strong correlations among {len(numeric_cols)} numeric columns"
    }

def generate_data_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive insights about the dataset."""
    insights = {
        'overview': {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'dtypes_summary': df.dtypes.value_counts().to_dict()
        },
        'missing_data': {
            'columns_with_missing': df.columns[df.isnull().any()].tolist(),
            'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'total_missing_cells': df.isnull().sum().sum()
        },
        'duplicates': {
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': df.duplicated().sum() / len(df) * 100
        }
    }
    
    # Numeric column insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        insights['numeric_summary'] = {
            'columns': numeric_cols.tolist(),
            'summary_stats': df[numeric_cols].describe().to_dict()
        }
    
    # Categorical column insights
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        insights['categorical_summary'] = {}
        for col in categorical_cols:
            unique_values = df[col].nunique()
            most_common = df[col].value_counts().head(3).to_dict()
            insights['categorical_summary'][col] = {
                'unique_values': unique_values,
                'most_common': most_common,
                'uniqueness_ratio': unique_values / len(df)
            }
    
    return insights

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality and provide recommendations."""
    issues = []
    recommendations = []
    score = 100  # Start with perfect score
    
    # Check missing data
    missing_percentage = df.isnull().sum().sum() / df.size * 100
    if missing_percentage > 20:
        issues.append(f"High missing data: {missing_percentage:.1f}% of cells are empty")
        recommendations.append("Consider data imputation or removing columns with excessive missing values")
        score -= 20
    elif missing_percentage > 5:
        issues.append(f"Moderate missing data: {missing_percentage:.1f}% of cells are empty")
        recommendations.append("Review missing data patterns and consider appropriate handling")
        score -= 10
    
    # Check duplicates
    duplicate_percentage = df.duplicated().sum() / len(df) * 100
    if duplicate_percentage > 10:
        issues.append(f"High duplicate rows: {duplicate_percentage:.1f}% of rows are duplicated")
        recommendations.append("Remove duplicate rows to improve data quality")
        score -= 15
    elif duplicate_percentage > 0:
        issues.append(f"Some duplicate rows: {duplicate_percentage:.1f}% of rows are duplicated")
        recommendations.append("Consider removing duplicate rows")
        score -= 5
    
    # Check data types
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        # Check if numeric data is stored as text
        try:
            pd.to_numeric(df[col], errors='raise')
            issues.append(f"Column '{col}' contains numeric data stored as text")
            recommendations.append(f"Convert column '{col}' to numeric type")
            score -= 5
        except:
            pass
    
    # Check for extremely skewed data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        skewness = abs(df[col].skew())
        if skewness > 2:
            issues.append(f"Column '{col}' is highly skewed (skewness: {skewness:.2f})")
            recommendations.append(f"Consider transforming '{col}' to reduce skewness")
            score -= 3
    
    return {
        'quality_score': max(0, score),
        'issues': issues,
        'recommendations': recommendations,
        'summary': f"Data quality score: {max(0, score)}/100"
    }

def create_executive_summary(insights: Dict[str, Any]) -> str:
    """Create an executive summary of data insights."""
    overview = insights.get('overview', {})
    missing = insights.get('missing_data', {})
    
    shape = overview.get('shape', [0, 0])
    missing_pct = (missing.get('total_missing_cells', 0) / (shape[0] * shape[1]) * 100) if shape[0] * shape[1] > 0 else 0
    
    summary = f"""
## Executive Data Summary

**Dataset Overview:**
- **Size:** {shape[0]:,} rows × {shape[1]} columns
- **Memory Usage:** {overview.get('memory_usage_mb', 0):.1f} MB
- **Data Completeness:** {100 - missing_pct:.1f}%

**Key Characteristics:**
- **Numeric Columns:** {len(insights.get('numeric_summary', {}).get('columns', []))}
- **Text Columns:** {len(insights.get('categorical_summary', {}))}
- **Missing Data:** {missing_pct:.1f}% of cells
- **Duplicate Rows:** {insights.get('duplicates', {}).get('duplicate_percentage', 0):.1f}%

**Data Quality Assessment:**
{insights.get('quality_summary', 'Assessment pending...')}
"""
    
    return summary.strip()

# Utility functions for specific data processing tasks
def safe_convert_to_numeric(series: pd.Series) -> pd.Series:
    """Safely convert a series to numeric, handling errors gracefully."""
    try:
        # First, try direct conversion
        return pd.to_numeric(series, errors='coerce')
    except:
        # If that fails, try cleaning the data first
        if series.dtype == 'object':
            # Remove common non-numeric characters
            cleaned = series.astype(str).str.replace(r'[,$%]', '', regex=True)
            return pd.to_numeric(cleaned, errors='coerce')
        return series

def parse_date_column(series: pd.Series) -> pd.Series:
    """Parse a series as dates with multiple format attempts."""
    if series.dtype == 'datetime64[ns]':
        return series
    
    # Common date formats to try
    date_formats = [
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S',
        '%m/%d/%Y %H:%M:%S', '%Y-%m-%d %H:%M', '%m/%d/%Y %H:%M'
    ]
    
    for fmt in date_formats:
        try:
            return pd.to_datetime(series, format=fmt, errors='raise')
        except:
            continue
    
    # If specific formats fail, try pandas' general parser
    try:
        return pd.to_datetime(series, errors='coerce')
    except:
        return series

def estimate_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """Estimate the most appropriate data type for each column."""
    recommendations = {}
    
    for col in df.columns:
        current_type = str(df[col].dtype)
        
        if current_type == 'object':
            # Try to determine if it should be numeric, date, or categorical
            non_null_values = df[col].dropna()
            
            if len(non_null_values) == 0:
                recommendations[col] = 'object'  # Keep as is if all null
                continue
            
            # Test for numeric
            try:
                pd.to_numeric(non_null_values.iloc[:100], errors='raise')  # Test first 100
                recommendations[col] = 'numeric'
                continue
            except:
                pass
            
            # Test for date
            try:
                pd.to_datetime(non_null_values.iloc[:100], errors='raise')  # Test first 100
                recommendations[col] = 'datetime'
                continue
            except:
                pass
            
            # Check if it should be categorical
            unique_ratio = non_null_values.nunique() / len(non_null_values)
            if unique_ratio < 0.1 and non_null_values.nunique() < 50:
                recommendations[col] = 'category'
            else:
                recommendations[col] = 'object'
        
        else:
            recommendations[col] = current_type
    
    return recommendations