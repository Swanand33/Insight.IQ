import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import io

class DataProcessor:
    """
    Core data processing engine for InsightGenie.
    Handles Excel/CSV parsing, data quality checks, and statistical analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data = None
        self.metadata = {}
        
    def load_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load Excel or CSV file and return processed data with metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict containing processed data and metadata
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path, engine='openpyxl')
            elif file_ext == '.csv':
                # Try different encodings and separators
                self.data = self._robust_csv_read(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
            # Generate metadata
            self.metadata = self._generate_metadata()
            
            return {
                'data': self.data,
                'metadata': self.metadata,
                'success': True,
                'message': f"Successfully loaded {len(self.data)} rows"
            }
            
        except Exception as e:
            self.logger.error(f"Error loading file: {str(e)}")
            return {
                'data': None,
                'metadata': {},
                'success': False,
                'message': f"Failed to load file: {str(e)}"
            }
    
    def _robust_csv_read(self, file_path: str) -> pd.DataFrame:
        """Attempt to read CSV with different encodings and separators."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        separators = [',', ';', '\t', '|']
        
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                    # Check if we got meaningful data (more than 1 column)
                    if len(df.columns) > 1:
                        return df
                except:
                    continue
        
        # Fallback to basic read
        return pd.read_csv(file_path)
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate comprehensive metadata about the dataset."""
        if self.data is None:
            return {}
        
        # Basic info
        metadata = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
        }
        
        # Data quality metrics
        metadata['data_quality'] = {
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'duplicate_rows': self.data.duplicated().sum(),
            'unique_values': {col: self.data[col].nunique() for col in self.data.columns}
        }
        
        # Column categorization
        metadata['column_types'] = self._categorize_columns()
        
        # Statistical summary for numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            metadata['statistics'] = self.data[numeric_cols].describe().to_dict()
        
        # Date columns analysis
        date_cols = self._identify_date_columns()
        if date_cols:
            metadata['date_analysis'] = self._analyze_date_columns(date_cols)
        
        return metadata
    
    def _categorize_columns(self) -> Dict[str, List[str]]:
        """Categorize columns by their data type and content."""
        categories = {
            'numeric': [],
            'categorical': [],
            'date': [],
            'text': [],
            'boolean': [],
            'mixed': []
        }
        
        for col in self.data.columns:
            dtype = str(self.data[col].dtype)
            unique_ratio = self.data[col].nunique() / len(self.data)
            
            if dtype in ['int64', 'float64', 'int32', 'float32']:
                categories['numeric'].append(col)
            elif dtype == 'bool':
                categories['boolean'].append(col)
            elif dtype == 'datetime64[ns]' or self._is_date_column(col):
                categories['date'].append(col)
            elif unique_ratio < 0.1 and self.data[col].nunique() < 50:
                categories['categorical'].append(col)
            elif dtype == 'object':
                if unique_ratio > 0.9:
                    categories['text'].append(col)
                else:
                    categories['mixed'].append(col)
        
        return categories
    
    def _identify_date_columns(self) -> List[str]:
        """Identify columns that contain date information."""
        date_columns = []
        
        for col in self.data.columns:
            if self._is_date_column(col):
                date_columns.append(col)
        
        return date_columns
    
    def _is_date_column(self, col: str) -> bool:
        """Check if a column contains date information."""
        # Check dtype first
        if self.data[col].dtype == 'datetime64[ns]':
            return True
        
        # Check column name patterns
        date_keywords = ['date', 'time', 'created', 'updated', 'timestamp', 'year', 'month', 'day']
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in date_keywords):
            # Try to parse a few values
            try:
                sample = self.data[col].dropna().head(10)
                parsed = pd.to_datetime(sample, errors='coerce')
                return parsed.notna().sum() > len(sample) * 0.5
            except:
                return False
        
        return False
    
    def _analyze_date_columns(self, date_cols: List[str]) -> Dict[str, Any]:
        """Analyze date columns for time-based insights."""
        analysis = {}
        
        for col in date_cols:
            try:
                # Convert to datetime if not already
                if self.data[col].dtype != 'datetime64[ns]':
                    date_series = pd.to_datetime(self.data[col], errors='coerce')
                else:
                    date_series = self.data[col]
                
                valid_dates = date_series.dropna()
                
                if len(valid_dates) > 0:
                    analysis[col] = {
                        'min_date': valid_dates.min(),
                        'max_date': valid_dates.max(),
                        'date_range_days': (valid_dates.max() - valid_dates.min()).days,
                        'missing_dates': len(date_series) - len(valid_dates),
                        'unique_dates': valid_dates.nunique()
                    }
            except:
                continue
        
        return analysis
    
    def get_key_insights(self) -> Dict[str, Any]:
        """Generate key insights from the data for AI analysis."""
        if self.data is None:
            return {}
        
        insights = {
            'dataset_summary': {
                'total_rows': len(self.data),
                'total_columns': len(self.data.columns),
                'data_quality_score': self._calculate_data_quality_score(),
                'primary_data_types': self._get_primary_data_types()
            },
            'numeric_insights': self._get_numeric_insights(),
            'categorical_insights': self._get_categorical_insights(),
            'temporal_insights': self._get_temporal_insights(),
            'anomalies': self._detect_anomalies()
        }
        
        return insights
    
    def _calculate_data_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)."""
        if self.data is None:
            return 0
        
        # Factors: missing values, duplicates, data type consistency
        missing_penalty = (self.data.isnull().sum().sum() / self.data.size) * 30
        duplicate_penalty = (self.data.duplicated().sum() / len(self.data)) * 20
        
        # Bonus for proper data types
        type_bonus = len(self.metadata['column_types']['numeric']) * 2
        
        score = max(0, 100 - missing_penalty - duplicate_penalty - type_bonus)
        return round(score, 1)
    
    def _get_primary_data_types(self) -> Dict[str, int]:
        """Get count of each primary data type."""
        return {k: len(v) for k, v in self.metadata['column_types'].items()}
    
    def _get_numeric_insights(self) -> Dict[str, Any]:
        """Extract insights from numeric columns."""
        numeric_cols = self.metadata['column_types']['numeric']
        if not numeric_cols:
            return {}
        
        insights = {}
        for col in numeric_cols:
            series = self.data[col]
            insights[col] = {
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'outliers': len(series[np.abs(series - series.mean()) > 2 * series.std()]),
                'trend': self._calculate_trend(series) if len(series) > 5 else 'insufficient_data'
            }
        
        return insights
    
    def _get_categorical_insights(self) -> Dict[str, Any]:
        """Extract insights from categorical columns."""
        categorical_cols = self.metadata['column_types']['categorical']
        if not categorical_cols:
            return {}
        
        insights = {}
        for col in categorical_cols:
            value_counts = self.data[col].value_counts()
            insights[col] = {
                'unique_values': len(value_counts),
                'most_common': value_counts.head(3).to_dict(),
                'distribution_evenness': self._calculate_distribution_evenness(value_counts)
            }
        
        return insights
    
    def _get_temporal_insights(self) -> Dict[str, Any]:
        """Extract insights from temporal columns."""
        date_cols = self.metadata['column_types']['date']
        if not date_cols:
            return {}
        
        insights = {}
        for col in date_cols:
            if col in self.metadata.get('date_analysis', {}):
                date_info = self.metadata['date_analysis'][col]
                insights[col] = {
                    'time_span_days': date_info['date_range_days'],
                    'data_frequency': self._estimate_data_frequency(col),
                    'seasonal_pattern': self._detect_seasonal_pattern(col)
                }
        
        return insights
    
    def _detect_anomalies(self) -> Dict[str, Any]:
        """Detect various types of anomalies in the data."""
        anomalies = {
            'statistical_outliers': {},
            'data_quality_issues': {},
            'pattern_breaks': {}
        }
        
        # Statistical outliers in numeric columns
        for col in self.metadata['column_types']['numeric']:
            series = self.data[col]
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = series[z_scores > 3]
            if len(outliers) > 0:
                anomalies['statistical_outliers'][col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(series) * 100,
                    'values': outliers.tolist()[:5]  # Show first 5
                }
        
        # Data quality issues
        for col in self.data.columns:
            missing_pct = self.data[col].isnull().sum() / len(self.data) * 100
            if missing_pct > 10:  # More than 10% missing
                anomalies['data_quality_issues'][col] = f"{missing_pct:.1f}% missing values"
        
        return anomalies
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction for a numeric series."""
        if len(series) < 2:
            return 'insufficient_data'
        
        # Simple linear trend
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_distribution_evenness(self, value_counts: pd.Series) -> str:
        """Calculate how evenly distributed categorical values are."""
        if len(value_counts) == 1:
            return 'single_value'
        
        # Calculate entropy-like measure
        probabilities = value_counts / value_counts.sum()
        evenness = -np.sum(probabilities * np.log2(probabilities))
        max_evenness = np.log2(len(value_counts))
        
        normalized_evenness = evenness / max_evenness if max_evenness > 0 else 0
        
        if normalized_evenness > 0.8:
            return 'highly_even'
        elif normalized_evenness > 0.5:
            return 'moderately_even'
        else:
            return 'skewed'
    
    def _estimate_data_frequency(self, date_col: str) -> str:
        """Estimate the frequency of data points in a date column."""
        if date_col not in self.data.columns:
            return 'unknown'
        
        date_series = pd.to_datetime(self.data[date_col], errors='coerce').dropna()
        if len(date_series) < 2:
            return 'insufficient_data'
        
        # Calculate median time difference
        sorted_dates = date_series.sort_values()
        diffs = sorted_dates.diff().dropna()
        median_diff = diffs.median()
        
        if median_diff.days >= 350:
            return 'yearly'
        elif median_diff.days >= 25:
            return 'monthly'
        elif median_diff.days >= 6:
            return 'weekly'
        elif median_diff.days >= 1:
            return 'daily'
        else:
            return 'sub_daily'
    
    def _detect_seasonal_pattern(self, date_col: str) -> str:
        """Detect if there's a seasonal pattern in the data."""
        # Simplified seasonal detection
        if date_col not in self.data.columns:
            return 'unknown'
        
        # This is a placeholder - would need more sophisticated analysis
        return 'analysis_needed'
    
    def export_summary(self) -> Dict[str, Any]:
        """Export a comprehensive summary for AI analysis."""
        return {
            'metadata': self.metadata,
            'insights': self.get_key_insights(),
            'sample_data': self.data.head(5).to_dict() if self.data is not None else {},
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate data processing recommendations."""
        recommendations = []
        
        if self.data is None:
            return recommendations
        
        # Missing data recommendations
        missing_cols = [col for col, pct in self.metadata['data_quality']['missing_percentage'].items() if pct > 10]
        if missing_cols:
            recommendations.append(f"Consider addressing missing values in: {', '.join(missing_cols)}")
        
        # Duplicate data
        if self.metadata['data_quality']['duplicate_rows'] > 0:
            recommendations.append("Remove duplicate rows to improve data quality")
        
        # Data type optimization
        if len(self.metadata['column_types']['mixed']) > 0:
            recommendations.append("Review mixed-type columns for potential data cleaning")
        
        return recommendations