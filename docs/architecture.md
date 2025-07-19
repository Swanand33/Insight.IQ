# InsightGenie Architecture

## System Overview

InsightGenie follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   File Upload   │────│   Data Processor │────│   AI Analyzer   │
│   (Gradio UI)   │    │   (Pandas Core)  │    │  (GPT-4 + Smart │
│                 │    │                  │    │     Prompts)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Auto-Visualizer │    │  Export Engine  │
                       │ (Plotly + Smart │    │ (HTML/MD/PDF)   │
                       │  Chart Selection)│    │                 │
                       └─────────────────┘    └─────────────────┘
```

## Core Components

### 1. Data Processor (`data_processor.py`)
- Handles Excel/CSV file parsing
- Performs data quality assessment
- Generates statistical insights
- Categorizes columns by type

### 2. AI Analyzer (`ai_analyzer.py`)
- Integrates with OpenAI GPT-4
- Uses specialized prompts for different analysis types
- Parses and structures AI responses
- Implements caching for cost optimization

### 3. Auto-Visualizer (`visualization.py`)
- Automatically selects appropriate chart types
- Generates interactive Plotly visualizations
- Creates comprehensive dashboards
- Exports to various formats

### 4. Main Application (`main.py`)
- Gradio-based user interface
- Orchestrates the analysis workflow
- Handles file uploads and downloads
- Manages user interactions

## Data Flow

1. **Upload**: User uploads Excel/CSV file
2. **Process**: Data is parsed and analyzed for quality
3. **Analyze**: AI generates business insights based on data patterns
4. **Visualize**: Charts are automatically generated based on data types
5. **Export**: Results are packaged for download

## Technology Stack

- **Frontend**: Gradio (Python-based web interface)
- **Data Processing**: Pandas, NumPy
- **AI Integration**: OpenAI GPT-4 API
- **Visualization**: Plotly, Matplotlib
- **Export**: HTML, Markdown, PDF