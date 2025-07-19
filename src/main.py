import gradio as gr
import pandas as pd
import os
import tempfile
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import json

from data_processor import DataProcessor
from ai_analyzer import AIAnalyzer, AnalysisResult
from visualization import AutoVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsightGenieApp:
    """Main application class for InsightGenie."""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.ai_analyzer = None
        self.visualizer = AutoVisualizer()
        self.current_data = None
        self.current_analysis = None
        self.current_dashboard = None
        
        # Initialize AI analyzer if API key is available
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.ai_analyzer = AIAnalyzer(api_key)
            logger.info("AI Analyzer initialized successfully")
        else:
            logger.warning("OPENAI_API_KEY not found. AI analysis will be disabled.")
    
    def process_file(self, file) -> Tuple[str, str, str]:
        """
        Process uploaded file and return results.
        
        Returns:
            Tuple of (status_message, data_preview, metadata_json)
        """
        try:
            if file is None:
                return "Please upload a file first.", "", ""
            
            # Save uploaded file to temporary location
            temp_path = self._save_uploaded_file(file)
            
            # Process the file
            result = self.data_processor.load_file(temp_path)
            
            if not result['success']:
                return f"Error: {result['message']}", "", ""
            
            # Store current data
            self.current_data = result
            
            # Generate preview
            preview = self._generate_data_preview(result['data'])
            
            # Generate metadata summary
            metadata_summary = self._format_metadata(result['metadata'])
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return result['message'], preview, metadata_summary
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return f"Error processing file: {str(e)}", "", ""
    
    def _save_uploaded_file(self, file) -> str:
        """Save uploaded file to temporary location."""
        # Create temp file with original extension
        original_name = file.name if hasattr(file, 'name') else 'uploaded_file'
        suffix = Path(original_name).suffix
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file.read())
            return tmp_file.name
    
    def _generate_data_preview(self, data: pd.DataFrame) -> str:
        """Generate a formatted preview of the data."""
        if data is None or data.empty:
            return "No data to preview"
        
        preview_parts = []
        
        # Basic info
        preview_parts.append(f"**Dataset Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
        preview_parts.append("")
        
        # Column info
        preview_parts.append("**Columns:**")
        for col in data.columns:
            dtype = str(data[col].dtype)
            non_null = data[col].count()
            preview_parts.append(f"- {col} ({dtype}) - {non_null} non-null values")
        
        preview_parts.append("")
        
        # First few rows
        preview_parts.append("**Sample Data (first 5 rows):**")
        preview_parts.append("```")
        preview_parts.append(data.head().to_string())
        preview_parts.append("```")
        
        return "\n".join(preview_parts)
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for display."""
        if not metadata:
            return "No metadata available"
        
        formatted = []
        
        # Data Quality Summary
        formatted.append("## 📊 Data Quality Summary")
        if 'data_quality' in metadata:
            dq = metadata['data_quality']
            total_missing = sum(dq['missing_values'].values())
            formatted.append(f"- **Missing Values:** {total_missing} total")
            formatted.append(f"- **Duplicate Rows:** {dq['duplicate_rows']}")
        
        formatted.append("")
        
        # Column Types
        formatted.append("## 🏷️ Column Types")
        if 'column_types' in metadata:
            for col_type, columns in metadata['column_types'].items():
                if columns:
                    formatted.append(f"- **{col_type.title()}:** {len(columns)} columns")
        
        formatted.append("")
        
        # Memory Usage
        if 'memory_usage' in metadata:
            memory_mb = metadata['memory_usage'] / (1024 * 1024)
            formatted.append(f"**Memory Usage:** {memory_mb:.2f} MB")
        
        return "\n".join(formatted)
    
    def analyze_data(self, analysis_type: str = "executive") -> Tuple[str, str]:
        """
        Perform AI analysis on the current data.
        
        Returns:
            Tuple of (analysis_result, confidence_info)
        """
        try:
            if self.current_data is None:
                return "Please upload and process a file first.", ""
            
            if self.ai_analyzer is None:
                return "AI analysis is not available. Please set your OPENAI_API_KEY environment variable.", ""
            
            # Get comprehensive data summary
            data_summary = self.data_processor.export_summary()
            
            # Perform AI analysis
            logger.info(f"Starting {analysis_type} analysis...")
            analysis_result = self.ai_analyzer.analyze_data(data_summary, analysis_type)
            
            # Store current analysis
            self.current_analysis = analysis_result
            
            # Format result for display
            formatted_result = self.ai_analyzer.format_result_for_display(analysis_result)
            
            # Generate confidence info
            confidence_info = self._format_confidence_info(analysis_result)
            
            return formatted_result, confidence_info
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
            return f"Error in AI analysis: {str(e)}", ""
    
    def generate_visualizations(self) -> Tuple[str, List[gr.Plot]]:
        """
        Generate visualizations for the current data.
        
        Returns:
            Tuple of (status_message, list_of_charts)
        """
        try:
            if self.current_data is None:
                return "Please upload and process a file first.", []
            
            logger.info("Generating visualizations...")
            
            # Generate dashboard
            dashboard = self.visualizer.generate_dashboard(
                self.current_data['data'], 
                self.current_data['metadata']
            )
            
            # Store current dashboard
            self.current_dashboard = dashboard
            
            # Convert charts to Gradio plots
            gradio_plots = []
            
            # Process all chart categories
            all_charts = (dashboard['overview_charts'] + 
                         dashboard['detailed_charts'] + 
                         dashboard['statistical_charts'])
            
            for chart in all_charts:
                if chart and chart.get('chart'):
                    try:
                        gradio_plots.append(chart['chart'])
                    except Exception as e:
                        logger.warning(f"Could not convert chart to Gradio: {e}")
                        continue
            
            # Generate summary
            summary = self.visualizer.create_executive_chart_summary(dashboard)
            
            success_msg = f"✅ Generated {len(gradio_plots)} visualizations successfully!"
            
            return success_msg, gradio_plots, summary
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return f"Error generating visualizations: {str(e)}", [], ""
    
    def export_dashboard(self) -> Optional[str]:
        """Export current dashboard to HTML file."""
        try:
            if self.current_dashboard is None:
                return None
            
            filename = f"dashboard_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
            exported_file = self.visualizer.export_charts_to_html(self.current_dashboard, filename)
            
            return exported_file
            
        except Exception as e:
            logger.error(f"Error exporting dashboard: {str(e)}")
            return None
        """Format confidence information for display."""
        confidence_pct = result.confidence_score * 100
        
        confidence_info = f"""
## 🎯 Analysis Quality
- **Confidence Score:** {confidence_pct:.1f}%
- **Analysis Type:** {result.analysis_type.title()}
- **Risk Level:** {result.risk_assessment}

## 📈 Insights Quality
- **Key Insights:** {len(result.key_insights)} items
- **Critical Issues:** {len(result.critical_issues)} items  
- **Recommendations:** {len(result.recommendations)} items
"""
        
        if confidence_pct >= 80:
            confidence_info += "\n✅ **High confidence** - Analysis is comprehensive and reliable"
        elif confidence_pct >= 60:
            confidence_info += "\n⚠️ **Medium confidence** - Analysis is good but may benefit from additional data"
        else:
            confidence_info += "\n🔴 **Low confidence** - Analysis may be limited by data quality or completeness"
        
        return confidence_info
    
    def export_analysis(self) -> Optional[str]:
        """Export current analysis to a downloadable format."""
        try:
            if self.current_analysis is None:
                return None
            
            # Create export content
            export_content = {
                'analysis_type': self.current_analysis.analysis_type,
                'timestamp': pd.Timestamp.now().isoformat(),
                'executive_summary': self.current_analysis.executive_summary,
                'key_insights': self.current_analysis.key_insights,
                'critical_issues': self.current_analysis.critical_issues,
                'recommendations': self.current_analysis.recommendations,
                'risk_assessment': self.current_analysis.risk_assessment,
                'confidence_score': self.current_analysis.confidence_score,
                'metadata': {
                    'dataset_shape': self.current_data['metadata']['shape'] if self.current_data else None,
                    'data_quality_score': self.data_processor.get_key_insights().get('dataset_summary', {}).get('data_quality_score', 'N/A')
                }
            }
            
            # Create formatted report
            formatted_report = self._create_formatted_report(export_content)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp_file:
                tmp_file.write(formatted_report)
                return tmp_file.name
            
        except Exception as e:
            logger.error(f"Error exporting analysis: {str(e)}")
            return None
    
    def _create_formatted_report(self, content: Dict[str, Any]) -> str:
        """Create a formatted markdown report."""
        report = f"""# 🧠 InsightGenie Analysis Report

**Generated:** {content['timestamp']}
**Analysis Type:** {content['analysis_type'].title()}
**Confidence Score:** {content['confidence_score']:.1%}

---

## 🎯 Executive Summary

{content['executive_summary']}

---

## 🔍 Key Insights

"""
        
        for i, insight in enumerate(content['key_insights'], 1):
            report += f"{i}. {insight}\n"
        
        report += """
---

## ⚠️ Critical Issues

"""
        
        for i, issue in enumerate(content['critical_issues'], 1):
            report += f"{i}. {issue}\n"
        
        report += """
---

## 📈 Strategic Recommendations

"""
        
        for i, rec in enumerate(content['recommendations'], 1):
            report += f"{i}. {rec}\n"
        
        report += f"""
---

## 🎲 Risk Assessment

**Overall Risk Level:** {content['risk_assessment']}

---

## 📋 Analysis Metadata

- **Dataset Shape:** {content['metadata']['dataset_shape']}
- **Data Quality Score:** {content['metadata']['data_quality_score']}/100
- **Analysis Confidence:** {content['confidence_score']:.1%}

---

*Report generated by InsightGenie - AI-Powered Excel Analytics*
"""
        
        return report

def create_app() -> gr.Blocks:
    """Create and configure the Gradio application."""
    app = InsightGenieApp()
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="InsightGenie - AI-Powered Excel Analytics",
        css="""
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .tab-nav {
            background-color: #f8f9fa;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🧠 InsightGenie</h1>
            <p>Transform spreadsheets into executive insights in seconds using AI</p>
        </div>
        """)
        
        with gr.Tabs(elem_classes="tab-nav") as tabs:
            
            # Tab 1: Data Upload and Processing
            with gr.Tab("📁 Upload & Process"):
                gr.Markdown("## Upload Your Excel or CSV File")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        file_input = gr.File(
                            label="Upload File",
                            file_types=[".xlsx", ".xls", ".csv"],
                            type="binary"
                        )
                        
                        process_btn = gr.Button(
                            "Process File", 
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        status_output = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=3
                        )
                
                with gr.Row():
                    with gr.Column():
                        data_preview = gr.Markdown(
                            label="Data Preview",
                            value="Upload a file to see preview..."
                        )
                    
                    with gr.Column():
                        metadata_output = gr.Markdown(
                            label="Data Summary",
                            value="Upload a file to see summary..."
                        )
            
            # Tab 2: AI Analysis
            with gr.Tab("🤖 AI Analysis"):
                gr.Markdown("## Generate AI-Powered Insights")
                
                with gr.Row():
                    analysis_type = gr.Dropdown(
                        choices=["executive", "financial", "operational", "generic"],
                        value="executive",
                        label="Analysis Type",
                        info="Choose the type of analysis to perform"
                    )
                    
                    analyze_btn = gr.Button(
                        "Generate Analysis",
                        variant="primary", 
                        size="lg"
                    )
                
                with gr.Row():
                    with gr.Column(scale=3):
                        analysis_output = gr.Markdown(
                            label="Analysis Results",
                            value="Process a file and click 'Generate Analysis' to see insights..."
                        )
                    
                    with gr.Column(scale=1):
                        confidence_output = gr.Markdown(
                            label="Analysis Quality",
                            value=""
                        )
                
                with gr.Row():
                    export_btn = gr.Button("📄 Export Report", variant="secondary")
                    download_file = gr.File(label="Download Report", visible=False)
            
            # Tab 3: Interactive Visualizations
            with gr.Tab("📊 Visualizations"):
                gr.Markdown("## Auto-Generated Charts & Dashboards")
                
                with gr.Row():
                    viz_btn = gr.Button(
                        "Generate Visualizations",
                        variant="primary",
                        size="lg"
                    )
                    export_dashboard_btn = gr.Button(
                        "📊 Export Dashboard",
                        variant="secondary"
                    )
                
                viz_status = gr.Textbox(
                    label="Visualization Status",
                    interactive=False,
                    lines=2
                )
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # Charts will be displayed here
                        chart_gallery = gr.Gallery(
                            label="Generated Charts",
                            show_label=True,
                            elem_id="chart_gallery",
                            columns=2,
                            height="auto"
                        )
                    
                    with gr.Column(scale=1):
                        viz_summary = gr.Markdown(
                            label="Chart Summary",
                            value="Generate visualizations to see summary..."
                        )
                
                dashboard_download = gr.File(label="Download Dashboard", visible=False)
            
            # Tab 4: Help & Info
            with gr.Tab("❓ Help"):
                gr.Markdown("""
                ## How to Use InsightGenie
                
                ### 1. Upload Your Data
                - Support formats: Excel (.xlsx, .xls) and CSV files
                - Maximum file size: 50MB
                - No preparation needed - upload your raw business data
                
                ### 2. Choose Analysis Type
                - **Executive**: Strategic insights for leadership
                - **Financial**: Revenue, costs, and financial metrics
                - **Operational**: Efficiency and performance analysis  
                - **Generic**: General data analysis for any dataset
                
                ### 3. Generate Visualizations
                - Click "Generate Visualizations" for auto-charts
                - View interactive plots and dashboards
                - Export complete dashboard as HTML
                - Multiple chart types: time series, distributions, correlations
                
                ### 4. Review Insights
                - Get executive summaries in business language
                - Identify key trends and patterns
                - Discover critical issues requiring attention
                - Receive actionable recommendations
                
                ### 5. Export Reports
                - Download professional markdown reports
                - Share insights with stakeholders
                - Use findings for decision-making
                
                ## Requirements
                - OpenAI API key set as environment variable `OPENAI_API_KEY`
                - Clean, structured data (Excel/CSV format)
                
                ## Tips for Best Results
                - Include column headers in your data
                - Use consistent date formats
                - Minimize missing values for better analysis
                - Include time-series data for trend analysis
                """)
        
        # Event handlers
        process_btn.click(
            fn=app.process_file,
            inputs=[file_input],
            outputs=[status_output, data_preview, metadata_output]
        )
        
        analyze_btn.click(
            fn=app.analyze_data,
            inputs=[analysis_type],
            outputs=[analysis_output, confidence_output]
        )
        
        viz_btn.click(
            fn=app.generate_visualizations,
            outputs=[viz_status, chart_gallery, viz_summary]
        )
        
        def handle_export():
            file_path = app.export_analysis()
            if file_path:
                return gr.update(value=file_path, visible=True)
            else:
                return gr.update(visible=False)
        
        export_btn.click(
            fn=handle_export,
            outputs=[download_file]
        )
        
        def handle_dashboard_export():
            file_path = app.export_dashboard()
            if file_path:
                return gr.update(value=file_path, visible=True)
            else:
                return gr.update(visible=False)
        
        export_dashboard_btn.click(
            fn=handle_dashboard_export,
            outputs=[dashboard_download]
        )
    
    return demo

def main():
    """Main application entry point."""
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set.")
        print("   AI analysis will be disabled. Set your API key to enable AI features.")
        print()
    
    # Create and launch app
    demo = create_app()
    
    print("🚀 Starting InsightGenie...")
    print("📊 Upload your Excel/CSV files to get started!")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True if you want to create a public link
        show_error=True
    )

if __name__ == "__main__":
    main()