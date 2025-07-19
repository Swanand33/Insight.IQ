import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

logger = logging.getLogger(__name__)

class AutoVisualizer:
    """
    Intelligent chart generation system that automatically selects
    the best visualizations based on data characteristics.
    """
    
    def __init__(self):
        self.color_palette = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#e67e22', '#34495e', '#e91e63', '#00bcd4'
        ]
        self.theme = {
            'layout': {
                'font': {'family': 'Arial, sans-serif', 'size': 12},
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'colorway': self.color_palette
            }
        }
    
    def generate_dashboard(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a complete dashboard with multiple charts.
        
        Args:
            data: DataFrame to visualize
            metadata: Data metadata from DataProcessor
            
        Returns:
            Dictionary containing all generated charts and summaries
        """
        try:
            dashboard = {
                'overview_charts': [],
                'detailed_charts': [],
                'statistical_charts': [],
                'summary': {},
                'recommendations': []
            }
            
            # Limit data size for visualization performance
            display_data = self._prepare_display_data(data)
            
            # Generate overview charts
            dashboard['overview_charts'] = self._create_overview_charts(display_data, metadata)
            
            # Generate detailed analysis charts
            dashboard['detailed_charts'] = self._create_detailed_charts(display_data, metadata)
            
            # Generate statistical visualizations
            dashboard['statistical_charts'] = self._create_statistical_charts(display_data, metadata)
            
            # Create summary
            dashboard['summary'] = self._create_visualization_summary(dashboard)
            
            # Generate recommendations
            dashboard['recommendations'] = self._generate_viz_recommendations(display_data, metadata)
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating dashboard: {str(e)}")
            return self._create_error_dashboard(str(e))
    
    def _prepare_display_data(self, data: pd.DataFrame, max_rows: int = 10000) -> pd.DataFrame:
        """Prepare data for visualization by sampling if necessary."""
        if len(data) <= max_rows:
            return data.copy()
        
        # Sample data intelligently
        # Keep first and last rows, sample middle
        sample_size = max_rows - 100  # Reserve 100 for first/last
        start_rows = data.head(50)
        end_rows = data.tail(50)
        
        if len(data) > 100:
            middle_sample = data.iloc[50:-50].sample(n=min(sample_size, len(data)-100))
            sampled_data = pd.concat([start_rows, middle_sample, end_rows]).sort_index()
        else:
            sampled_data = data
        
        logger.info(f"Sampled {len(sampled_data)} rows from {len(data)} for visualization")
        return sampled_data
    
    def _create_overview_charts(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create high-level overview charts."""
        charts = []
        
        # Data Quality Overview
        charts.append(self._create_data_quality_chart(data, metadata))
        
        # Column Type Distribution
        charts.append(self._create_column_type_chart(metadata))
        
        # Missing Data Heatmap
        if data.isnull().sum().sum() > 0:
            charts.append(self._create_missing_data_chart(data))
        
        # Numeric Columns Summary
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            charts.append(self._create_numeric_summary_chart(data[numeric_cols]))
        
        return [chart for chart in charts if chart is not None]
    
    def _create_detailed_charts(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create detailed analysis charts."""
        charts = []
        
        # Time series charts
        date_cols = self._identify_date_columns(data)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if date_cols and len(numeric_cols) > 0:
            for date_col in date_cols[:2]:  # Limit to 2 date columns
                for numeric_col in numeric_cols[:3]:  # Limit to 3 numeric columns
                    chart = self._create_time_series_chart(data, date_col, numeric_col)
                    if chart:
                        charts.append(chart)
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            charts.append(self._create_correlation_heatmap(data[numeric_cols]))
        
        # Distribution charts for key numeric columns
        for col in numeric_cols[:4]:  # Limit to top 4 numeric columns
            charts.append(self._create_distribution_chart(data[col], col))
        
        # Categorical analysis
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols[:3]:  # Limit to 3 categorical columns
            if data[col].nunique() <= 20:  # Only for reasonable number of categories
                charts.append(self._create_categorical_chart(data[col], col))
        
        return [chart for chart in charts if chart is not None]
    
    def _create_statistical_charts(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create statistical analysis charts."""
        charts = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Box plots for outlier detection
        if len(numeric_cols) > 0:
            charts.append(self._create_box_plot_chart(data[numeric_cols]))
        
        # Scatter plot matrix for correlations
        if len(numeric_cols) >= 2:
            top_numeric = numeric_cols[:4]  # Limit to 4 for readability
            charts.append(self._create_scatter_matrix(data[top_numeric]))
        
        # Statistical summary table
        if len(numeric_cols) > 0:
            charts.append(self._create_statistical_summary_table(data[numeric_cols]))
        
        return [chart for chart in charts if chart is not None]
    
    def _create_data_quality_chart(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create data quality overview chart."""
        try:
            # Calculate quality metrics
            total_cells = data.size
            missing_cells = data.isnull().sum().sum()
            complete_cells = total_cells - missing_cells
            duplicate_rows = data.duplicated().sum()
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Complete Data', 'Missing Data'],
                values=[complete_cells, missing_cells],
                hole=0.4,
                marker_colors=['#2ecc71', '#e74c3c']
            )])
            
            fig.update_layout(
                title="Data Completeness Overview",
                annotations=[dict(text=f'{(complete_cells/total_cells)*100:.1f}%<br>Complete', 
                                x=0.5, y=0.5, font_size=16, showarrow=False)],
                **self.theme['layout']
            )
            
            return {
                'type': 'overview',
                'title': 'Data Quality Overview',
                'chart': fig,
                'insights': [
                    f"Data is {(complete_cells/total_cells)*100:.1f}% complete",
                    f"{missing_cells:,} missing values out of {total_cells:,} total cells",
                    f"{duplicate_rows} duplicate rows found" if duplicate_rows > 0 else "No duplicate rows"
                ]
            }
        except Exception as e:
            logger.error(f"Error creating data quality chart: {e}")
            return None
    
    def _create_column_type_chart(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create column type distribution chart."""
        try:
            column_types = metadata.get('column_types', {})
            
            # Count non-empty lists
            type_counts = {k: len(v) for k, v in column_types.items() if len(v) > 0}
            
            if not type_counts:
                return None
            
            fig = go.Figure(data=[go.Bar(
                x=list(type_counts.keys()),
                y=list(type_counts.values()),
                marker_color=self.color_palette[:len(type_counts)]
            )])
            
            fig.update_layout(
                title="Column Types Distribution",
                xaxis_title="Data Type",
                yaxis_title="Number of Columns",
                **self.theme['layout']
            )
            
            return {
                'type': 'overview',
                'title': 'Column Types',
                'chart': fig,
                'insights': [f"{count} {dtype} columns" for dtype, count in type_counts.items()]
            }
        except Exception as e:
            logger.error(f"Error creating column type chart: {e}")
            return None
    
    def _create_missing_data_chart(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create missing data visualization."""
        try:
            missing_data = data.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
            
            if len(missing_data) == 0:
                return None
            
            fig = go.Figure(data=[go.Bar(
                x=missing_data.values,
                y=missing_data.index,
                orientation='h',
                marker_color='#e74c3c'
            )])
            
            fig.update_layout(
                title="Missing Data by Column",
                xaxis_title="Number of Missing Values",
                yaxis_title="Columns",
                height=max(400, len(missing_data) * 25),
                **self.theme['layout']
            )
            
            return {
                'type': 'overview',
                'title': 'Missing Data Analysis',
                'chart': fig,
                'insights': [f"{col}: {count} missing values" for col, count in missing_data.head(5).items()]
            }
        except Exception as e:
            logger.error(f"Error creating missing data chart: {e}")
            return None
    
    def _create_numeric_summary_chart(self, numeric_data: pd.DataFrame) -> Dict[str, Any]:
        """Create numeric columns summary chart."""
        try:
            # Calculate statistics
            stats = numeric_data.describe()
            
            # Create subplot for mean values
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=stats.columns,
                y=stats.loc['mean'],
                name='Mean Values',
                marker_color='#3498db'
            ))
            
            fig.update_layout(
                title="Numeric Columns - Mean Values",
                xaxis_title="Columns",
                yaxis_title="Mean Value",
                xaxis_tickangle=-45,
                **self.theme['layout']
            )
            
            return {
                'type': 'overview',
                'title': 'Numeric Summary',
                'chart': fig,
                'insights': [f"{col}: Mean = {stats.loc['mean', col]:.2f}" for col in stats.columns[:5]]
            }
        except Exception as e:
            logger.error(f"Error creating numeric summary chart: {e}")
            return None
    
    def _create_time_series_chart(self, data: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """Create time series chart."""
        try:
            # Prepare data
            plot_data = data[[date_col, value_col]].copy()
            plot_data[date_col] = pd.to_datetime(plot_data[date_col], errors='coerce')
            plot_data = plot_data.dropna().sort_values(date_col)
            
            if len(plot_data) < 2:
                return None
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=plot_data[date_col],
                y=plot_data[value_col],
                mode='lines+markers',
                name=value_col,
                line=dict(color='#3498db', width=2),
                marker=dict(size=4)
            ))
            
            # Add trend line
            if len(plot_data) > 3:
                z = np.polyfit(range(len(plot_data)), plot_data[value_col], 1)
                trend_line = np.poly1d(z)(range(len(plot_data)))
                
                fig.add_trace(go.Scatter(
                    x=plot_data[date_col],
                    y=trend_line,
                    mode='lines',
                    name='Trend',
                    line=dict(color='#e74c3c', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=f"{value_col} Over Time",
                xaxis_title=date_col,
                yaxis_title=value_col,
                **self.theme['layout']
            )
            
            # Calculate trend
            if len(plot_data) > 1:
                trend_direction = "increasing" if plot_data[value_col].iloc[-1] > plot_data[value_col].iloc[0] else "decreasing"
                trend_pct = ((plot_data[value_col].iloc[-1] - plot_data[value_col].iloc[0]) / plot_data[value_col].iloc[0] * 100)
            else:
                trend_direction = "stable"
                trend_pct = 0
            
            return {
                'type': 'detailed',
                'title': f'{value_col} Time Series',
                'chart': fig,
                'insights': [
                    f"Trend: {trend_direction} ({trend_pct:+.1f}%)",
                    f"Time span: {plot_data[date_col].min().strftime('%Y-%m-%d')} to {plot_data[date_col].max().strftime('%Y-%m-%d')}",
                    f"Data points: {len(plot_data)}"
                ]
            }
        except Exception as e:
            logger.error(f"Error creating time series chart: {e}")
            return None
    
    def _create_correlation_heatmap(self, numeric_data: pd.DataFrame) -> Dict[str, Any]:
        """Create correlation heatmap."""
        try:
            corr_matrix = numeric_data.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Correlation Matrix",
                width=600,
                height=600,
                **self.theme['layout']
            )
            
            # Find strongest correlations
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        correlations.append(
                            f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.2f}"
                        )
            
            return {
                'type': 'detailed',
                'title': 'Correlation Analysis',
                'chart': fig,
                'insights': correlations[:5] if correlations else ["No strong correlations found"]
            }
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return None
    
    def _create_distribution_chart(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Create distribution chart for a numeric column."""
        try:
            clean_data = series.dropna()
            
            if len(clean_data) == 0:
                return None
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[f'{column_name} Distribution', f'{column_name} Box Plot'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=clean_data, nbinsx=30, name='Distribution', marker_color='#3498db'),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=clean_data, name='Box Plot', marker_color='#e74c3c'),
                row=1, col=2
            )
            
            fig.update_layout(
                title=f"{column_name} - Distribution Analysis",
                showlegend=False,
                **self.theme['layout']
            )
            
            # Calculate insights
            insights = [
                f"Mean: {clean_data.mean():.2f}",
                f"Median: {clean_data.median():.2f}",
                f"Std Dev: {clean_data.std():.2f}",
                f"Range: {clean_data.min():.2f} to {clean_data.max():.2f}"
            ]
            
            return {
                'type': 'detailed',
                'title': f'{column_name} Distribution',
                'chart': fig,
                'insights': insights
            }
        except Exception as e:
            logger.error(f"Error creating distribution chart: {e}")
            return None
    
    def _create_categorical_chart(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Create chart for categorical data."""
        try:
            value_counts = series.value_counts().head(15)  # Top 15 categories
            
            if len(value_counts) == 0:
                return None
            
            # Choose chart type based on number of categories
            if len(value_counts) <= 7:
                # Pie chart for few categories
                fig = go.Figure(data=[go.Pie(
                    labels=value_counts.index,
                    values=value_counts.values,
                    marker_colors=self.color_palette[:len(value_counts)]
                )])
                chart_type = "Distribution"
            else:
                # Bar chart for many categories
                fig = go.Figure(data=[go.Bar(
                    x=value_counts.values,
                    y=value_counts.index,
                    orientation='h',
                    marker_color=self.color_palette[0]
                )])
                fig.update_layout(
                    xaxis_title="Count",
                    yaxis_title=column_name
                )
                chart_type = "Top Categories"
            
            fig.update_layout(
                title=f"{column_name} - {chart_type}",
                **self.theme['layout']
            )
            
            insights = [
                f"Unique categories: {series.nunique()}",
                f"Most common: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)",
                f"Distribution evenness: {'Even' if value_counts.std() < value_counts.mean() else 'Skewed'}"
            ]
            
            return {
                'type': 'detailed',
                'title': f'{column_name} Analysis',
                'chart': fig,
                'insights': insights
            }
        except Exception as e:
            logger.error(f"Error creating categorical chart: {e}")
            return None
    
    def _create_box_plot_chart(self, numeric_data: pd.DataFrame) -> Dict[str, Any]:
        """Create box plots for outlier detection."""
        try:
            if len(numeric_data.columns) == 0:
                return None
            
            fig = go.Figure()
            
            for i, col in enumerate(numeric_data.columns[:6]):  # Limit to 6 columns
                fig.add_trace(go.Box(
                    y=numeric_data[col],
                    name=col,
                    marker_color=self.color_palette[i % len(self.color_palette)]
                ))
            
            fig.update_layout(
                title="Box Plots - Outlier Detection",
                yaxis_title="Values",
                **self.theme['layout']
            )
            
            # Calculate outlier insights
            outlier_counts = {}
            for col in numeric_data.columns[:6]:
                Q1 = numeric_data[col].quantile(0.25)
                Q3 = numeric_data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = numeric_data[col][(numeric_data[col] < Q1 - 1.5*IQR) | (numeric_data[col] > Q3 + 1.5*IQR)]
                outlier_counts[col] = len(outliers)
            
            insights = [f"{col}: {count} outliers" for col, count in outlier_counts.items() if count > 0]
            
            return {
                'type': 'statistical',
                'title': 'Outlier Analysis',
                'chart': fig,
                'insights': insights if insights else ["No significant outliers detected"]
            }
        except Exception as e:
            logger.error(f"Error creating box plot chart: {e}")
            return None
    
    def _create_scatter_matrix(self, numeric_data: pd.DataFrame) -> Dict[str, Any]:
        """Create scatter plot matrix."""
        try:
            if len(numeric_data.columns) < 2:
                return None
            
            # Use plotly express for scatter matrix
            fig = px.scatter_matrix(
                numeric_data.sample(min(1000, len(numeric_data))),  # Sample for performance
                dimensions=numeric_data.columns[:4],  # Limit to 4 dimensions
                color_discrete_sequence=[self.color_palette[0]]
            )
            
            fig.update_layout(
                title="Scatter Plot Matrix - Variable Relationships",
                **self.theme['layout']
            )
            
            # Calculate correlation insights
            corr_matrix = numeric_data.corr()
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.6:
                        strong_corrs.append(
                            f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: {corr_val:.2f}"
                        )
            
            return {
                'type': 'statistical',
                'title': 'Variable Relationships',
                'chart': fig,
                'insights': strong_corrs[:3] if strong_corrs else ["No strong relationships found"]
            }
        except Exception as e:
            logger.error(f"Error creating scatter matrix: {e}")
            return None
    
    def _create_statistical_summary_table(self, numeric_data: pd.DataFrame) -> Dict[str, Any]:
        """Create statistical summary table."""
        try:
            stats = numeric_data.describe().round(2)
            
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Statistic'] + list(stats.columns),
                    fill_color='#3498db',
                    font=dict(color='white', size=12),
                    align='left'
                ),
                cells=dict(
                    values=[stats.index] + [stats[col] for col in stats.columns],
                    fill_color='#ecf0f1',
                    align='left',
                    font=dict(size=11)
                )
            )])
            
            fig.update_layout(
                title="Statistical Summary Table",
                **self.theme['layout']
            )
            
            insights = [
                f"Analyzed {len(numeric_data.columns)} numeric columns",
                f"Dataset contains {len(numeric_data)} records",
                "Key statistics: mean, median, std deviation calculated"
            ]
            
            return {
                'type': 'statistical',
                'title': 'Statistical Summary',
                'chart': fig,
                'insights': insights
            }
        except Exception as e:
            logger.error(f"Error creating statistical summary table: {e}")
            return None
    
    def _identify_date_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify columns that contain date information."""
        date_columns = []
        
        for col in data.columns:
            # Check dtype first
            if data[col].dtype == 'datetime64[ns]':
                date_columns.append(col)
                continue
            
            # Check column name patterns
            date_keywords = ['date', 'time', 'created', 'updated', 'timestamp', 'year', 'month', 'day']
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in date_keywords):
                # Try to parse a sample
                try:
                    sample = data[col].dropna().head(10)
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.notna().sum() > len(sample) * 0.7:  # 70% success rate
                        date_columns.append(col)
                except:
                    continue
        
        return date_columns
    
    def _create_visualization_summary(self, dashboard: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of all visualizations."""
        total_charts = (len(dashboard['overview_charts']) + 
                       len(dashboard['detailed_charts']) + 
                       len(dashboard['statistical_charts']))
        
        chart_types = []
        if dashboard['overview_charts']:
            chart_types.append(f"{len(dashboard['overview_charts'])} overview charts")
        if dashboard['detailed_charts']:
            chart_types.append(f"{len(dashboard['detailed_charts'])} detailed charts")
        if dashboard['statistical_charts']:
            chart_types.append(f"{len(dashboard['statistical_charts'])} statistical charts")
        
        return {
            'total_charts': total_charts,
            'chart_breakdown': chart_types,
            'summary_text': f"Generated {total_charts} charts: {', '.join(chart_types)}"
        }
    
    def _generate_viz_recommendations(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> List[str]:
        """Generate recommendations for better visualizations."""
        recommendations = []
        
        # Check data size
        if len(data) > 50000:
            recommendations.append("Consider filtering data for better chart performance")
        
        # Check missing data
        missing_pct = data.isnull().sum().sum() / data.size * 100
        if missing_pct > 20:
            recommendations.append("High missing data may affect chart accuracy - consider data cleaning")
        
        # Check numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 10:
            recommendations.append("Many numeric columns detected - consider focusing on key metrics")
        
        # Check categorical data
        categorical_cols = data.select_dtypes(include=['object']).columns
        high_cardinality = [col for col in categorical_cols if data[col].nunique() > 50]
        if high_cardinality:
            recommendations.append(f"High cardinality in {len(high_cardinality)} columns - consider grouping categories")
        
        # Check time series potential
        date_cols = self._identify_date_columns(data)
        if date_cols and len(numeric_cols) > 0:
            recommendations.append("Time series analysis possible - consider trend analysis")
        
        return recommendations
    
    def _create_error_dashboard(self, error_message: str) -> Dict[str, Any]:
        """Create error dashboard when visualization fails."""
        return {
            'overview_charts': [],
            'detailed_charts': [],
            'statistical_charts': [],
            'summary': {
                'total_charts': 0,
                'chart_breakdown': [],
                'summary_text': f"Visualization failed: {error_message}"
            },
            'recommendations': ["Check data format and try again"]
        }
    
    def create_executive_chart_summary(self, dashboard: Dict[str, Any]) -> str:
        """Create an executive summary of all charts."""
        summary = "# 📊 Executive Dashboard Summary\n\n"
        
        # Overview
        total_charts = dashboard['summary']['total_charts']
        summary += f"**Generated {total_charts} visualizations** covering data overview, detailed analysis, and statistical insights.\n\n"
        
        # Key findings from charts
        summary += "## 🔍 Key Visual Insights\n\n"
        
        all_insights = []
        for chart_group in ['overview_charts', 'detailed_charts', 'statistical_charts']:
            for chart in dashboard.get(chart_group, []):
                if chart.get('insights'):
                    all_insights.extend(chart['insights'])
        
        # Display top insights
        for i, insight in enumerate(all_insights[:10], 1):
            summary += f"{i}. {insight}\n"
        
        # Recommendations
        if dashboard.get('recommendations'):
            summary += "\n## 💡 Visualization Recommendations\n\n"
            for i, rec in enumerate(dashboard['recommendations'], 1):
                summary += f"{i}. {rec}\n"
        
        return summary