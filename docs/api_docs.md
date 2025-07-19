# InsightGenie API Documentation

## Core Classes and Methods

### DataProcessor

Main class for handling data processing and analysis.

#### Methods

##### `load_file(file_path: str) -> Dict[str, Any]`
Loads and processes Excel or CSV files.

**Parameters:**
- `file_path`: Path to the file to process

**Returns:**
- Dictionary with keys: `success`, `data`, `metadata`, `message`

##### `get_key_insights() -> Dict[str, Any]`
Generates key insights from the loaded data.

**Returns:**
- Dictionary containing dataset summary, numeric insights, categorical insights, and anomalies

---

### AIAnalyzer

Handles AI-powered analysis using GPT-4.

#### Methods

##### `analyze_data(data_summary: Dict[str, Any], analysis_type: str) -> AnalysisResult`
Performs AI analysis on data summary.

**Parameters:**
- `data_summary`: Processed data summary from DataProcessor
- `analysis_type`: Type of analysis ('executive', 'financial', 'operational', 'generic')

**Returns:**
- `AnalysisResult` object with structured insights

---

### AutoVisualizer

Generates automatic visualizations and dashboards.

#### Methods

##### `generate_dashboard(data: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]`
Creates a complete dashboard with multiple charts.

**Parameters:**
- `data`: DataFrame to visualize
- `metadata`: Data metadata from DataProcessor

**Returns:**
- Dictionary containing overview charts, detailed charts, statistical charts, and summary

---

## Usage Examples

### Basic Data Processing
```python
from data_processor import DataProcessor

processor = DataProcessor()
result = processor.load_file('data.xlsx')
if result['success']:
    insights = processor.get_key_insights()
```

### AI Analysis
```python
from ai_analyzer import AIAnalyzer

analyzer = AIAnalyzer(api_key="your-key")
analysis = analyzer.analyze_data(data_summary, 'executive')
```

### Visualization
```python
from visualization import AutoVisualizer

visualizer = AutoVisualizer()
dashboard = visualizer.generate_dashboard(data, metadata)
```